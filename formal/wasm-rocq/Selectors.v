(* Opcode-selector Boolean and one-hot facts from the exported R1CS. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range OpcodeCatalog.
Import ListNotations.
Open Scope Z_scope.

(* Keep large row indices compact during elaboration on the LSP's small worker stack. *)
Local Number Notation nat Nat.of_num_uint Nat.to_num_uint (abstract after 1000) : nat_scope.

Definition selectorColumns : list Column :=
  seq 61%nat 85%nat ++ seq 293%nat 30%nat.

Definition selectorRow : Row :=
  (map (fun column => (column, 1)) selectorColumns ++
     [(COL_IS_PROGRAM_ROW, -1)], [(COL_ONE, 1)], []).

Lemma selector_row_in_full :
  nth_error FullRelation.rows 379%nat = Some selectorRow.
Proof. vm_compute. reflexivity. Qed.

Lemma program_row_boolean_in_full :
  nth_error FullRelation.rows 1006%nat =
    Some (Range.booleanRow COL_IS_PROGRAM_ROW).
Proof. vm_compute. reflexivity. Qed.

Lemma selector_boolean_blocks :
  firstn 85%nat (skipn 1589%nat FullRelation.rows) =
    map Range.booleanRow (seq 61%nat 85%nat) /\
  firstn 30%nat (skipn 3430%nat FullRelation.rows) =
    map Range.booleanRow (seq 293%nat 30%nat).
Proof. vm_compute. split; reflexivity. Qed.

Definition selectorSum (w : Witness) (columns : list Column) : Z :=
  fold_right (fun column sum => w column + sum) 0 columns.

Lemma selector_sum_nonnegative : forall (w : Witness) columns,
  (forall column, In column columns -> 0 <= w column) ->
  0 <= selectorSum w columns.
Proof.
  intros w columns Hnonnegative.
  revert Hnonnegative.
  induction columns; intros Hnonnegative; cbn [selectorSum fold_right].
  - lia.
  - pose proof (value_bound w a) as [Hlo _].
    assert (Hrest : forall c, In c columns -> 0 <= w c).
    { intros c Hin. apply Hnonnegative. right. exact Hin. }
    specialize (IHcolumns Hrest). unfold selectorSum in IHcolumns. lia.
Qed.

Lemma selector_sum_active : forall (w : Witness) columns,
  Forall (fun column => w column = 0 \/ w column = 1) columns ->
  selectorSum w columns = 1 ->
  exists column, In column columns /\ w column = 1.
Proof.
  intros w columns Hbits. induction Hbits as [|column columns Hbit Hbits IH].
  - cbn. discriminate.
  - change (w column + selectorSum w columns = 1 ->
      exists chosen, In chosen (column :: columns) /\ w chosen = 1).
    intros Hsum. destruct Hbit as [Hzero | Hone].
    + destruct (IH ltac:(lia)) as [chosen [Hin Hchosen]].
      exists chosen. split; [right; exact Hin | exact Hchosen].
    + exists column. split; [left; reflexivity | exact Hone].
Qed.

Lemma selector_sum_one : forall (w : Witness) columns column,
  In column columns ->
  w column <= selectorSum w columns.
Proof.
  intros w columns.
  induction columns; intros column Hin.
  - contradiction.
  - cbn [selectorSum fold_right].
    destruct Hin as [Heq | Hin].
    + subst. pose proof (selector_sum_nonnegative w columns
        (fun c _ => proj1 (value_bound w c))) as Hrest.
      unfold selectorSum in Hrest. lia.
    + specialize (IHcolumns column Hin).
      unfold selectorSum in IHcolumns.
      pose proof (value_bound w a) as [Hhead _]. lia.
Qed.

Lemma selector_sum_two : forall (w : Witness) columns first second,
  first <> second -> In first columns -> In second columns ->
  w first + w second <= selectorSum w columns.
Proof.
  intros w columns.
  induction columns; intros first second Hneq Hfirst Hsecond.
  - contradiction.
  - cbn [selectorSum fold_right].
    destruct Hfirst as [Hfirst | Hfirst];
      destruct Hsecond as [Hsecond | Hsecond].
    + subst. contradiction.
    + subst. pose proof (selector_sum_one w columns second Hsecond) as Hsub.
      unfold selectorSum in Hsub. lia.
    + subst. pose proof (selector_sum_one w columns first Hfirst) as Hsub.
      unfold selectorSum in Hsub. lia.
    + specialize (IHcolumns first second Hneq Hfirst Hsecond).
      unfold selectorSum in IHcolumns.
      pose proof (value_bound w a) as [Hhead _]. lia.
Qed.

Lemma selector_sum_eval : forall (w : Witness) columns,
  eval w (map (fun column => (column, 1)) columns) =
    selectorSum w columns.
Proof.
  intros w columns. induction columns.
  - reflexivity.
  - change (1 * w a + eval w (map (fun column => (column, 1)) columns) =
      w a + selectorSum w columns).
    rewrite IHcolumns. ring.
Qed.

Lemma eval_app : forall w first second,
  eval w (first ++ second) = eval w first + eval w second.
Proof.
  intros w first second. induction first.
  - reflexivity.
  - destruct a as [column coefficient].
    change (coefficient * w column + eval w (first ++ second) =
      coefficient * w column + eval w first + eval w second).
    rewrite IHfirst. ring.
Qed.

Lemma selector_sum_boolean_bound : forall (w : Witness) columns,
  Forall (fun column => w column = 0 \/ w column = 1) columns ->
  0 <= selectorSum w columns <= Z.of_nat (length columns).
Proof.
  intros w columns Hbits. induction Hbits.
  - cbn. lia.
  - cbn [selectorSum fold_right length Z.of_nat].
    unfold selectorSum in IHHbits.
    destruct H; subst; lia.
Qed.

Lemma selector_boolean_from_block : forall (w : Witness) start count columns,
  Range.fieldPrime -> FullRelation.satFull w ->
  firstn count (skipn start FullRelation.rows) =
    map Range.booleanRow columns ->
  Forall (fun column => w column = 0 \/ w column = 1) columns.
Proof.
  intros w start count columns Hprime [Hone Hall] Hblock.
  apply Forall_forall. intros column Hin.
  apply Range.boolean_row_bound; try assumption.
  pose proof (Range.sat_slice w start count Hall) as Hslice.
  rewrite Hblock in Hslice.
  apply (proj1 (Forall_forall _ _) Hslice).
  apply in_map. exact Hin.
Qed.

Theorem full_opcode_selectors_one_hot : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  Forall (fun column => w column = 0 \/ w column = 1) selectorColumns /\
  selectorSum w selectorColumns = w COL_IS_PROGRAM_ROW /\
  (w COL_IS_PROGRAM_ROW = 0 \/ w COL_IS_PROGRAM_ROW = 1).
Proof.
  intros w Hprime Hfull.
  destruct Hfull as [Hone Hall].
  destruct selector_boolean_blocks as [HblockA HblockB].
  pose proof (selector_boolean_from_block w 1589%nat 85%nat
    (seq 61%nat 85%nat) Hprime (conj Hone Hall) HblockA) as HA.
  pose proof (selector_boolean_from_block w 3430%nat 30%nat
    (seq 293%nat 30%nat) Hprime (conj Hone Hall) HblockB) as HB.
  assert (Hbits : Forall (fun c => w c = 0 \/ w c = 1) selectorColumns).
  { unfold selectorColumns. apply Forall_app. split; assumption. }
  pose proof (selector_sum_boolean_bound w selectorColumns Hbits) as HsumBound.
  change (0 <= selectorSum w selectorColumns <= 115) in HsumBound.
  pose proof (Range.boolean_row_bound w COL_IS_PROGRAM_ROW Hprime Hone
    (Range.sat_row w 1006%nat _ Hall program_row_boolean_in_full)) as Hprogram.
  pose proof (Range.sat_row w 379%nat _ Hall selector_row_in_full) as Honehot.
  unfold sat, selectorRow in Honehot.
  rewrite eval_app in Honehot.
  rewrite selector_sum_eval in Honehot.
  cbn [eval fold_right] in Honehot.
  change (w COL_ONE = 1) in Hone.
  rewrite Hone in Honehot.
  assert (Hsum : selectorSum w selectorColumns = w COL_IS_PROGRAM_ROW).
  { apply Z.mod_divide in Honehot; try (unfold modulus; lia).
    destruct Honehot as [k Hk].
    destruct Hprogram; unfold modulus in *; lia. }
  repeat split; assumption.
Qed.

Lemma full_selected_is_program_row : forall (w : Witness) selector,
  Range.fieldPrime -> FullRelation.satFull w ->
  In selector selectorColumns -> w selector = 1 ->
  w COL_IS_PROGRAM_ROW = 1.
Proof.
  intros w selector Hprime Hfull Hin Hselected.
  pose proof (full_opcode_selectors_one_hot w Hprime Hfull)
    as [_ [Hsum Hprogram]].
  pose proof (selector_sum_one w selectorColumns selector Hin) as Hbound.
  rewrite Hselected, Hsum in Hbound.
  destruct Hprogram; lia.
Qed.

Lemma full_opcode_selector_exclusive : forall (w : Witness) chosen other,
  Range.fieldPrime -> FullRelation.satFull w ->
  w chosen = 1 ->
  In chosen selectorColumns -> In other selectorColumns -> chosen <> other ->
  w other = 0.
Proof.
  intros w chosen other Hprime Hfull Hchosen HchosenIn HotherIn Hneq.
  pose proof (full_opcode_selectors_one_hot w Hprime Hfull)
    as [_ [Hsum Hprogram]].
  pose proof (selector_sum_two w selectorColumns chosen other
    Hneq HchosenIn HotherIn) as Htwo.
  pose proof (value_bound w other) as [Hlo _].
  rewrite Hchosen, Hsum in Htwo.
  destruct Hprogram; lia.
Qed.

(* The decoder's coefficients come from the same Rust opcode catalog as the ROM. *)
Definition opcodeDecodeRow : Row :=
  ((COL_OPCODE_CODE, 1) :: map (fun '(column, code) => (column, -code)) selectorCodes,
   [(COL_ONE, 1)], []).

Lemma opcode_decode_in_full :
  nth_error FullRelation.rows 420%nat = Some opcodeDecodeRow.
Proof. vm_compute. reflexivity. Qed.

Lemma opcode_selector_columns : forall column,
  In column (map fst selectorCodes) <-> In column selectorColumns.
Proof.
  assert (Hforward : forallb (fun c => existsb (Nat.eqb c) selectorColumns)
    (map fst selectorCodes) = true) by (vm_compute; reflexivity).
  assert (Hbackward : forallb (fun c => existsb (Nat.eqb c) (map fst selectorCodes))
    selectorColumns = true) by (vm_compute; reflexivity).
  intro column. split; intro Hin.
  - apply forallb_forall with (x := column) in Hforward; [ | exact Hin].
    apply existsb_exists in Hforward. destruct Hforward as [c [Hc Heq]].
    apply Nat.eqb_eq in Heq. subst. exact Hc.
  - apply forallb_forall with (x := column) in Hbackward; [ | exact Hin].
    apply existsb_exists in Hbackward. destruct Hbackward as [c [Hc Heq]].
    apply Nat.eqb_eq in Heq. subst. exact Hc.
Qed.

Lemma opcode_selector_columns_unique : NoDup (map fst selectorCodes).
Proof.
  replace (map fst selectorCodes) with (nodup Nat.eq_dec (map fst selectorCodes))
    by (vm_compute; reflexivity).
  apply NoDup_nodup.
Qed.

Lemma opcode_codes_bounded :
  Forall (fun entry => 0 <= snd entry < modulus) selectorCodes.
Proof.
  assert (Hcheck : forallb (fun entry => (0 <=? snd entry) && (snd entry <? modulus))
    selectorCodes = true) by (vm_compute; reflexivity).
  apply Forall_forall. intros entry Hin.
  apply forallb_forall with (x := entry) in Hcheck; [ | exact Hin].
  apply Bool.andb_true_iff in Hcheck. destruct Hcheck as [Hlo Hhi].
  apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi. split; assumption.
Qed.

Lemma eval_zero_terms : forall (w : Witness) terms,
  (forall column, In column (map fst terms) -> w column = 0) -> eval w terms = 0.
Proof.
  intros w terms. induction terms as [|[column coefficient] rest IH]; intros Hzero.
  - reflexivity.
  - change (coefficient * w column + eval w rest = 0).
    rewrite Hzero by (left; reflexivity).
    rewrite IH by (intros c Hin; apply Hzero; right; exact Hin). ring.
Qed.

Lemma eval_single_term : forall (w : Witness) terms chosen code,
  NoDup (map fst terms) -> In (chosen, code) terms -> w chosen = 1 ->
  (forall column, In column (map fst terms) -> column <> chosen -> w column = 0) ->
  eval w terms = code.
Proof.
  intros w terms. induction terms as [|[column coefficient] rest IH];
    intros chosen code Hunique Hin Hone Hzero; [contradiction | ].
  inversion Hunique as [|head tail Habsent Htail]; subst.
  destruct Hin as [Heq | Hin].
  - inversion Heq; subst. change (code * w chosen + eval w rest = code). rewrite Hone.
    rewrite eval_zero_terms; [ring | ].
    intros c Hc. apply Hzero; [right; exact Hc | ].
    intro Hsame. subst c. contradiction.
  - assert (Hneq : column <> chosen).
    { intro Heq. subst column. apply Habsent.
      change (In (fst (chosen, code)) (map fst rest)). apply in_map. exact Hin. }
    change (coefficient * w column + eval w rest = code).
    rewrite Hzero by (try (left; reflexivity); exact Hneq).
    rewrite (IH chosen code Htail Hin Hone); [ring | ].
    intros c Hc Hneq'. apply Hzero; [right; exact Hc | exact Hneq'].
Qed.

Lemma eval_negated_terms : forall (w : Witness) terms,
  eval w (map (fun '(column, coefficient) => (column, -coefficient)) terms) =
  - eval w terms.
Proof.
  intros w terms. induction terms as [|[column coefficient] rest IH].
  - reflexivity.
  - change (-coefficient * w column +
      eval w (map (fun '(c, k) => (c, -k)) rest) =
      -(coefficient * w column + eval w rest)).
    rewrite IH. ring.
Qed.

Theorem full_opcode_decode : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  exists selector code,
    In (selector, code) selectorCodes /\ w selector = 1 /\ w COL_OPCODE_CODE = code.
Proof.
  intros w Hprime Hfull Hprogram.
  pose proof (full_opcode_selectors_one_hot w Hprime Hfull) as [Hbits [Hsum _]].
  rewrite Hprogram in Hsum.
  destruct (selector_sum_active w selectorColumns Hbits Hsum) as [chosen [Hin Hone]].
  rewrite <- opcode_selector_columns in Hin.
  apply in_map_iff in Hin. destruct Hin as [[selector code] [Heq Hentry]].
  cbn in Heq. subst chosen.
  assert (Hselector : In selector selectorColumns).
  { rewrite <- opcode_selector_columns. change (In (fst (selector, code)) (map fst selectorCodes)).
    apply in_map. exact Hentry. }
  assert (Heval : eval w selectorCodes = code).
  { apply eval_single_term with (chosen := selector); try assumption.
    - exact opcode_selector_columns_unique.
    - intros other Hother Hneq.
      rewrite opcode_selector_columns in Hother.
      eapply full_opcode_selector_exclusive; try eassumption. congruence. }
  pose proof (Range.sat_row w 420%nat _ (proj2 Hfull) opcode_decode_in_full) as Hdecode.
  unfold sat, opcodeDecodeRow in Hdecode.
  change (((1 * w COL_OPCODE_CODE +
    eval w (map (fun '(column, code) => (column, -code)) selectorCodes)) *
    (1 * w COL_ONE + 0) - 0) mod modulus = 0) in Hdecode.
  assert (Hconst : w COL_ONE = 1) by exact (proj1 Hfull).
  rewrite eval_negated_terms, Heval, Hconst in Hdecode.
  pose proof (value_bound w COL_OPCODE_CODE) as Hbound.
  pose proof (proj1 (Forall_forall _ _) opcode_codes_bounded (selector, code) Hentry) as Hcode.
  cbn in Hcode.
  apply Z.mod_divide in Hdecode; [ | unfold modulus; lia].
  destruct Hdecode as [k Hk].
  exists selector, code. repeat split; try assumption. unfold modulus in *; lia.
Qed.

Theorem full_opcode_selects : forall (w : Witness) selector code,
  Range.fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  filter (fun entry => Z.eqb (snd entry) code) selectorCodes = [(selector, code)] ->
  w COL_OPCODE_CODE = code -> w selector = 1.
Proof.
  intros w selector code Hprime Hfull Hprogram Hunique Hcode.
  destruct (full_opcode_decode w Hprime Hfull Hprogram) as [chosen [actual [Hin [Hchosen Hactual]]]].
  assert (Hfiltered : In (chosen, actual)
    (filter (fun entry => Z.eqb (snd entry) code) selectorCodes)).
  { apply filter_In. split; [exact Hin | cbn; apply Z.eqb_eq; congruence]. }
  rewrite Hunique in Hfiltered. destruct Hfiltered as [Heq | Hfalse]; [ | contradiction].
  inversion Heq; subst. exact Hchosen.
Qed.

Definition rowKindRow : Row :=
  ([(COL_IS_PROGRAM_ROW, 1); (COL_PARAM_INIT_ACTIVE_BEFORE, 1);
    (COL_LOCAL_ZERO_ACTIVE_BEFORE, 1); (COL_TAIL_ENTER_ACTIVE, 1);
    (COL_PADDING_ACTIVE, 1); (COL_GATHER_ACTIVE, 1);
    (COL_TURN_BOUNDARY, 1); (COL_PERM_PENDING_BEFORE, 1);
    (COL_PERM_ROUND_BEFORE_IS_ZERO, -1)],
   [(COL_ONE, 1)], []).

Lemma row_kind_row_in_full :
  nth_error FullRelation.rows 37%nat = Some rowKindRow.
Proof. vm_compute. reflexivity. Qed.

Lemma full_boolean_at : forall (w : Witness) column index,
  Range.fieldPrime -> FullRelation.satFull w ->
  nth_error FullRelation.rows index = Some (Range.booleanRow column) ->
  0 <= w column <= 1.
Proof.
  intros w column index Hprime [Hone Hall] Hrow.
  pose proof (Range.boolean_row_bound w column Hprime Hone
    (Range.sat_row w index _ Hall Hrow)) as Hbit.
  destruct Hbit; lia.
Qed.

Lemma full_opcode_aux_gates_zero : forall (w : Witness) selected,
  Range.fieldPrime -> FullRelation.satFull w ->
  In selected selectorColumns -> w selected = 1 ->
  w COL_PARAM_INIT_ACTIVE_BEFORE = 0 /\
  w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0 /\
  w COL_GATHER_ACTIVE = 0 /\ w COL_TAIL_ENTER_ACTIVE = 0.
Proof.
  intros w selected Hprime Hfull HselectedIn HselectedValue.
  pose proof (full_opcode_selectors_one_hot w Hprime Hfull)
    as [_ [Hsum HprogramBit]].
  pose proof (selector_sum_one w selectorColumns
    selected HselectedIn) as Hselected.
  rewrite HselectedValue, Hsum in Hselected.
  assert (Hprogram : w COL_IS_PROGRAM_ROW = 1).
  { destruct HprogramBit; lia. }
  destruct Hfull as [Hone Hall].
  pose proof (full_boolean_at w COL_PARAM_INIT_ACTIVE_BEFORE 1078%nat
    Hprime (conj Hone Hall) eq_refl) as Hparam.
  pose proof (full_boolean_at w COL_LOCAL_ZERO_ACTIVE_BEFORE 1009%nat
    Hprime (conj Hone Hall) eq_refl) as Hzero.
  pose proof (full_boolean_at w COL_TAIL_ENTER_ACTIVE 1082%nat
    Hprime (conj Hone Hall) eq_refl) as Htail.
  pose proof (full_boolean_at w COL_PADDING_ACTIVE 1116%nat
    Hprime (conj Hone Hall) eq_refl) as Hpadding.
  pose proof (full_boolean_at w COL_GATHER_ACTIVE 3651%nat
    Hprime (conj Hone Hall) eq_refl) as Hgather.
  pose proof (full_boolean_at w COL_TURN_BOUNDARY 3657%nat
    Hprime (conj Hone Hall) eq_refl) as Hturn.
  pose proof (full_boolean_at w COL_PERM_PENDING_BEFORE 3648%nat
    Hprime (conj Hone Hall) eq_refl) as Hpending.
  pose proof (full_boolean_at w COL_PERM_ROUND_BEFORE_IS_ZERO 3650%nat
    Hprime (conj Hone Hall) eq_refl) as Hround.
  pose proof (Range.sat_row w 37%nat _ Hall row_kind_row_in_full) as Hrow.
  unfold sat, rowKindRow in Hrow.
  cbn [eval fold_right] in Hrow.
  change (w COL_ONE = 1) in Hone.
  rewrite Hone, Hprogram in Hrow.
  apply Z.mod_divide in Hrow; try (unfold modulus; lia).
  destruct Hrow as [k Hk].
  unfold modulus in *; lia.
Qed.

Definition gatherKindColumns : list Column := seq 534%nat 10%nat.

Definition gatherKindRow : Row :=
  (map (fun column => (column, 1)) gatherKindColumns ++
    [(COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], []).

Lemma gather_kind_row_in_full :
  nth_error FullRelation.rows 154%nat = Some gatherKindRow.
Proof. vm_compute. reflexivity. Qed.

Lemma gather_kind_boolean_block :
  firstn 10%nat (skipn 3687%nat FullRelation.rows) =
    map Range.booleanRow gatherKindColumns.
Proof. vm_compute. reflexivity. Qed.

Lemma full_no_gather_kind : forall (w : Witness) column,
  Range.fieldPrime -> FullRelation.satFull w -> w COL_GATHER_ACTIVE = 0 ->
  In column gatherKindColumns -> w column = 0.
Proof.
  intros w column Hprime Hfull Hgather Hin.
  pose proof (selector_boolean_from_block w 3687%nat 10%nat
    gatherKindColumns Hprime Hfull gather_kind_boolean_block) as Hbits.
  pose proof (selector_sum_boolean_bound w gatherKindColumns Hbits) as Hbound.
  change (0 <= selectorSum w gatherKindColumns <= 10) in Hbound.
  pose proof (Range.sat_row w 154%nat _ (proj2 Hfull) gather_kind_row_in_full) as Hrow.
  unfold sat, gatherKindRow in Hrow.
  rewrite eval_app, selector_sum_eval in Hrow. cbn [eval fold_right] in Hrow.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  rewrite Hone, Hgather in Hrow.
  pose proof (selector_sum_one w gatherKindColumns column Hin) as Hcol.
  pose proof (value_bound w column) as Hvalue.
  apply Z.mod_divide in Hrow; [ | unfold modulus; lia].
  destruct Hrow as [k Hk]. unfold modulus in *; lia.
Qed.

(* Substitute the known one-hot selector values before interpreting a row. *)
Definition selectedTerms (chosen : Column) (terms : SparseRow) : SparseRow :=
  flat_map (fun '(column, coefficient) =>
    if existsb (Nat.eqb column) selectorColumns then
      if Nat.eqb column chosen then [(COL_ONE, coefficient)] else []
    else [(column, coefficient)]) terms.

Definition selectedRow (chosen : Column) (row : Row) : Row :=
  let '(a, b, c) := row in
  (selectedTerms chosen a, selectedTerms chosen b, selectedTerms chosen c).

Lemma selected_terms_eval : forall (w : Witness) chosen terms,
  Range.fieldPrime -> FullRelation.satFull w ->
  In chosen selectorColumns -> w chosen = 1 ->
  eval w (selectedTerms chosen terms) = eval w terms.
Proof.
  intros w chosen terms Hprime Hfull Hin Hone.
  induction terms as [|[column coefficient] rest IH]; [reflexivity | ].
  unfold selectedTerms at 1. cbn [flat_map]. fold (selectedTerms chosen rest).
  rewrite eval_app, IH.
  destruct (existsb (Nat.eqb column) selectorColumns) eqn:Hcol.
  - apply existsb_exists in Hcol. destruct Hcol as [c [Hc Heq]].
    apply Nat.eqb_eq in Heq. subst c.
    destruct (Nat.eqb column chosen) eqn:Hsame.
    + apply Nat.eqb_eq in Hsame. subst column.
      change (coefficient * w COL_ONE + 0 + eval w rest = coefficient * w chosen + eval w rest).
      assert (Hconst : w COL_ONE = 1) by exact (proj1 Hfull).
      rewrite Hconst, Hone. ring.
    + apply Nat.eqb_neq in Hsame.
      pose proof (full_opcode_selector_exclusive w chosen column Hprime Hfull Hone Hin Hc
        ltac:(congruence)) as Hzero.
      change (0 + eval w rest = coefficient * w column + eval w rest).
      rewrite Hzero. ring.
  - change (coefficient * w column + 0 + eval w rest = coefficient * w column + eval w rest). ring.
Qed.

Lemma full_selected_row : forall (w : Witness) chosen index row,
  Range.fieldPrime -> FullRelation.satFull w ->
  In chosen selectorColumns -> w chosen = 1 ->
  nth_error FullRelation.rows index = Some row -> sat w (selectedRow chosen row).
Proof.
  intros w chosen index [[a b] c] Hprime Hfull Hin Hone Hrow.
  pose proof (Range.sat_row w index _ (proj2 Hfull) Hrow) as Hsat.
  unfold sat, selectedRow in *. repeat rewrite selected_terms_eval by assumption. exact Hsat.
Qed.

Ltac normalize_columns :=
  repeat match goal with
  | H : context [value ?w ?column] |- _ =>
      let n := eval compute in column in
      tryif constr_eq column n then fail else
        change (value w column) with (value w n) in H
  | |- context [value ?w ?column] =>
      let n := eval compute in column in
      tryif constr_eq column n then fail else
        change (value w column) with (value w n)
  end.

Ltac field_linear H :=
  normalize_columns;
  apply Z.mod_divide in H; [ | unfold modulus; lia ];
  let k := fresh "multiple" in destruct H as [k H]; unfold modulus in *; lia.

(* Rewrite every gather-kind column in H to zero, given Hzero over gatherKindColumns. *)
Ltac zero_gather_kinds Hzero H :=
  repeat match type of H with
  | context [value ?w ?column] =>
      let member := eval vm_compute in (existsb (Nat.eqb column) gatherKindColumns) in
      lazymatch member with
      | true => rewrite (Hzero column ltac:(cbn; tauto)) in H
      end
  end.

Local Ltac full_row index name :=
  match goal with
  | Hfull : FullRelation.satFull ?w |- _ =>
  let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
  let Hlookup := fresh "Hlookup" in
  assert (Hlookup : nth_error FullRelation.rows index = Some raw) by (vm_compute; reflexivity);
  pose proof (Range.sat_row w index raw (proj2 Hfull) Hlookup) as name; clear Hlookup;
  unfold sat in name; cbn [eval fold_right] in name; normalize_columns
  end.

(* Gather and tail-entry terms appear in rows shared by every row kind. *)
Record ProgramRowQuiet (w : Witness) : Prop := {
  quiet_gather_kinds : forall column, In column gatherKindColumns -> w column = 0;
  quiet_gather_high : w GK2_HI = 0;
  quiet_gather_local : w GMEM_LOCAL = 0;
  quiet_gather_output : w GMEM_OUTPUT = 0;
  quiet_tail_discard : w COL_TAIL_DISCARD_COUNT = 0
}.

Theorem full_program_row_quiet : forall (w : Witness) selected,
  Range.fieldPrime -> FullRelation.satFull w ->
  In selected selectorColumns -> w selected = 1 -> ProgramRowQuiet w.
Proof.
  intros w selected Hprime Hfull Hin Hselected.
  pose proof (full_opcode_aux_gates_zero w selected Hprime Hfull Hin Hselected)
    as [_ [_ [Hgather Htail]]].
  assert (Hkinds : forall column, In column gatherKindColumns -> w column = 0)
    by (intros; eapply full_no_gather_kind; eassumption).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hhigh : w GK2_HI = 0).
  { full_row gatherHighLimbRow Hrow. zero_gather_kinds Hkinds Hrow.
    pose proof (value_bound w GK2_HI). field_linear Hrow. }
  assert (Hgmem : w GMEM_LOCAL = 0 /\ w GMEM_OUTPUT = 0).
  { full_row gatherMemoryKindRow Hrow. zero_gather_kinds Hkinds Hrow.
    pose proof (full_boolean_at w GMEM_LOCAL gmemLocalBooleanRow Hprime Hfull eq_refl).
    pose proof (full_boolean_at w GMEM_BYTE gmemByteBooleanRow Hprime Hfull eq_refl).
    pose proof (full_boolean_at w GMEM_HALF gmemHalfBooleanRow Hprime Hfull eq_refl).
    pose proof (full_boolean_at w GMEM_OUTPUT gmemOutputBooleanRow Hprime Hfull eq_refl).
    field_linear Hrow. }
  assert (Hdiscard : w COL_TAIL_DISCARD_COUNT = 0).
  { full_row tailDiscardInactiveRow Hrow. rewrite Hone, Htail in Hrow.
    pose proof (value_bound w COL_TAIL_DISCARD_COUNT). field_linear Hrow. }
  destruct Hgmem. constructor; assumption.
Qed.

Ltac selected_row chosen index name :=
  match goal with
  | Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
    let reduced := eval vm_compute in (selectedRow chosen raw) in
    pose proof (full_selected_row w chosen index raw Hprime Hfull ltac:(assumption) ltac:(assumption) eq_refl) as name;
    change (sat w reduced) in name;
    unfold sat in name; cbn [eval fold_right] in name; normalize_columns
  end.
