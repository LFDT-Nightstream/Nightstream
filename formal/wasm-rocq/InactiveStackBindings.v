From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog InactiveOffset.
Import ListNotations.
Open Scope Z_scope.

(* SPEC.md section 7.6: private advice may be zeroed when its consumers are off. *)
Definition zeroColumn (column : nat) (w : Witness) : Witness :=
 {| entries := fun c => if Nat.eq_dec c column
    then exist (fun z : Z => 0 <= z < modulus) 0 ltac:(unfold modulus; lia) else entries w c |}.
Lemma zero_column_same : forall column w c, c <> column -> zeroColumn column w c = w c.
Proof.
 intros column w c Hc. unfold value, zeroColumn. cbn.
 destruct (Nat.eq_dec c column); [contradiction|reflexivity].
Qed.
Lemma zero_column_value : forall column w, zeroColumn column w column = 0.
Proof.
 intros column w. unfold value, zeroColumn.
 cbv beta iota zeta delta [entries]. destruct (Nat.eq_dec column column); [reflexivity|contradiction].
Qed.

Definition privateRowSafe (column : nat) (gates : list SparseRow) (row : Row) : bool :=
 let '(a,b,c) := row in
 (omitsColumn column a && omitsColumn column b && omitsColumn column c) ||
 (omitsColumn column a && termsEqual c [] && existsb (termsEqual a) gates) ||
 (termsEqual a [(column,1)] && termsEqual c []).

Lemma private_row_repair : forall column gates (w : Witness) row,
 Forall (fun gate => eval w gate = 0) gates ->
 privateRowSafe column gates row = true -> sat w row -> sat (zeroColumn column w) row.
Proof.
 intros column gates w [[a b] c] Hgates Hsafe Hsat.
 unfold privateRowSafe in Hsafe.
 apply orb_true_iff in Hsafe as [Hsafe|Hsafe].
 - apply orb_true_iff in Hsafe as [Hsafe|Hsafe].
   + apply andb_true_iff in Hsafe as [Hab Hc].
     apply andb_true_iff in Hab as [Ha Hb].
     unfold sat in *.
     rewrite (eval_omitted_column w (zeroColumn column w) column a (zero_column_same column w) Ha).
     rewrite (eval_omitted_column w (zeroColumn column w) column b (zero_column_same column w) Hb).
     rewrite (eval_omitted_column w (zeroColumn column w) column c (zero_column_same column w) Hc).
     exact Hsat.
   + apply andb_true_iff in Hsafe as [Hac Hmember].
     apply andb_true_iff in Hac as [Ha Hempty].
     apply terms_equal_sound in Hempty. subst c.
     apply existsb_exists in Hmember as [gate [Hin Heq]].
     apply terms_equal_sound in Heq. subst gate.
     pose proof (proj1 (Forall_forall _ _) Hgates a Hin) as Hz.
     unfold sat. rewrite (eval_omitted_column w (zeroColumn column w) column a (zero_column_same column w) Ha).
     rewrite Hz. reflexivity.
 - apply andb_true_iff in Hsafe as [Ha Hc].
   apply terms_equal_sound in Ha. apply terms_equal_sound in Hc. subst a c.
   unfold sat. cbn [eval fold_right]. rewrite zero_column_value. reflexivity.
Qed.

Lemma private_full_repair : forall column gates (w : Witness),
 column <> COL_ONE ->
 forallb (privateRowSafe column gates) FullRelation.rows = true ->
 Forall (fun gate => eval w gate = 0) gates ->
 FullRelation.satFull w -> FullRelation.satFull (zeroColumn column w).
Proof.
 intros column gates w Hcolumn Hsafe Hgates Hfull. split.
 - change (zeroColumn column w COL_ONE = 1).
   rewrite zero_column_same by congruence. exact (proj1 Hfull).
 - apply Forall_forall. intros row Hin.
   eapply private_row_repair; [exact Hgates| |].
   + exact (proj1 (forallb_forall _ _) Hsafe row Hin).
   + exact (proj1 (Forall_forall _ _) (proj2 Hfull) row Hin).
Qed.

Definition globalGates : list SparseRow := [[(COL_SEL_GLOBAL_GET,1)]; [(COL_SEL_GLOBAL_SET,1)]].
Definition tableIndexGates : list SparseRow :=
 [[(COL_SEL_TABLE_GET,1)]; [(COL_SEL_TABLE_SET,1)]; [(COL_SEL_CALL_INDIRECT_RESOLVE,1)];
  [(COL_SEL_TABLE_GET,1); (COL_SEL_TABLE_SET,1); (COL_SEL_CALL_INDIRECT_RESOLVE,1)]].

(* An out-of-range value cannot equal a bounded stack operand at an active gate. *)
Lemma bound_or_gate_zero : forall (w : Witness) column stack gate,
 fieldPrime ->
 4294967296 <= w column -> 0 <= w stack < 4294967296 ->
 (w gate = 0 \/ w gate = 1) ->
 sat w ([(gate,1)], [(column,1); (stack,-1)], []) -> w gate = 0.
Proof.
 intros w column stack gate Hprime Hlarge Hstack Hgate Hsat.
 destruct Hgate as [Hgate|Hgate]; [exact Hgate|].
 pose proof (value_bound w column).
 pose proof (sat_nonzero_right w [(gate,1)] [(column,1);(stack,-1)] Hprime Hsat) as Hz.
 cbn [eval fold_right] in Hz. rewrite Hgate in Hz. unfold modulus in *; lia.
Qed.

Ltac inactive_binding column stack gate :=
 match goal with Hfull : FullRelation.satFull ?w |- _ =>
 let Hrange := fresh "Hrange" in let Hbool := fresh "Hbool" in let Hrow := fresh "Hrow" in
 circuit_range stack Hrange; circuit_boolean gate Hbool;
 circuit_row ([(gate,1)], [(column,1);(stack,-1)], ([] : SparseRow)) Hrow;
 let Hz := fresh "Hz" in
 assert (Hz : w gate = 0) by (eapply bound_or_gate_zero; eauto; lia);
 clear Hrange Hbool Hrow
 end.

Lemma global_lo_inactive : forall w,
 fieldPrime -> FullRelation.satFull w -> 4294967296 <= w COL_GLOBAL_VALUE ->
 w COL_SEL_GLOBAL_GET = 0 /\ w COL_SEL_GLOBAL_SET = 0.
Proof.
 intros w Hprime Hfull Hlarge.
 inactive_binding COL_GLOBAL_VALUE COL_STACK_WRITE0_VALUE_LO COL_SEL_GLOBAL_GET.
 inactive_binding COL_GLOBAL_VALUE (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) COL_SEL_GLOBAL_SET.
 auto.
Qed.
Lemma global_hi_inactive : forall w,
 fieldPrime -> FullRelation.satFull w -> 4294967296 <= w COL_GLOBAL_VALUE_HI ->
 w COL_SEL_GLOBAL_GET = 0 /\ w COL_SEL_GLOBAL_SET = 0.
Proof.
 intros w Hprime Hfull Hlarge.
 inactive_binding COL_GLOBAL_VALUE_HI COL_STACK_WRITE0_VALUE_HI COL_SEL_GLOBAL_GET.
 inactive_binding COL_GLOBAL_VALUE_HI (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat) COL_SEL_GLOBAL_SET.
 auto.
Qed.
Lemma table_index_inactive : forall w,
 fieldPrime -> FullRelation.satFull w -> 4294967296 <= w COL_TABLE_INDEX ->
 w COL_SEL_TABLE_GET = 0 /\ w COL_SEL_TABLE_SET = 0 /\
 w COL_SEL_CALL_INDIRECT_RESOLVE = 0 /\ w COL_TABLE_READ_ENABLED = 0 /\ w COL_TABLE_SET_LIVE = 0.
Proof.
 intros w Hprime Hfull Hlarge.
 inactive_binding COL_TABLE_INDEX (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) COL_SEL_TABLE_GET.
 inactive_binding COL_TABLE_INDEX (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) COL_SEL_TABLE_SET.
 inactive_binding COL_TABLE_INDEX (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) COL_SEL_CALL_INDIRECT_RESOLVE.
 circuit_boolean COL_CI_OOB Bci. circuit_boolean COL_TABLE_READ_ENABLED Bread.
 circuit_boolean COL_TABLE_SET_LIVE Bwrite.
 circuit_row ([(COL_SEL_TABLE_GET,1)], [(COL_TABLE_OOB,1)],
   [(COL_SEL_TABLE_GET,1); (COL_SEL_CALL_INDIRECT_RESOLVE,1); (COL_CI_OOB,-1); (COL_TABLE_READ_ENABLED,-1)]) Hr.
 circuit_row ([(COL_SEL_TABLE_SET,1)], [(COL_ONE,1);(COL_CMP_GE,-1)], [(COL_TABLE_SET_LIVE,1)]) Hw.
 expand_row Hr. expand_row Hw.
 repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr, Hw end.
 apply Z.mod_divide in Hr; [|unfold modulus;lia].
 apply Z.mod_divide in Hw; [|unfold modulus;lia].
 destruct Hr as [kr Hr]. destruct Hw as [kw Hw]. unfold modulus in *; intuition nia.
Qed.

Lemma global_lo_rows_safe : forallb (privateRowSafe COL_GLOBAL_VALUE globalGates) FullRelation.rows = true.
Proof. vm_compute. reflexivity. Qed.

Theorem full_global_lo_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> exists w',
 FullRelation.satFull w' /\
 0 <= w' COL_GLOBAL_VALUE < 4294967296 /\
 eventsForRow memories w' = eventsForRow memories w /\
 map (fun '(a,b) => (w' a,w' b)) continuityLinks =
 map (fun '(a,b) => (w a,w b)) continuityLinks /\
 (forall c, c <> COL_GLOBAL_VALUE -> w' c = w c).
Proof.
 intros w Hprime Hfull.
 destruct (Z_lt_ge_dec (w COL_GLOBAL_VALUE) 4294967296) as [Hsmall|Hlarge].
 - exists w. split; [exact Hfull|]. split; [pose proof (value_bound w COL_GLOBAL_VALUE);lia|].
   repeat split; reflexivity.
 - destruct (global_lo_inactive w Hprime Hfull ltac:(lia)) as [Hget Hset].
   exists (zeroColumn COL_GLOBAL_VALUE w). split.
   + eapply private_full_repair with (gates := globalGates).
     * vm_compute; discriminate.
     * exact global_lo_rows_safe.
     * unfold globalGates. repeat constructor; cbn [eval fold_right]; rewrite ?Hget, ?Hset; reflexivity.
     * exact Hfull.
   + split; [rewrite zero_column_value; lia|]. split.
     * cbv beta iota zeta delta [eventsForRow memories flat_map ports memoryId portEvent active activation addressColumns valueColumn portKind map option_map].
       repeat match goal with
       | |- context [value (zeroColumn ?column ?witness) ?c] =>
           let Hneq := fresh in
           assert (Hneq : c <> column) by (vm_compute; discriminate);
           rewrite (zero_column_same column witness c Hneq); clear Hneq
       end.
       normalize_columns. rewrite Hget, Hset. reflexivity.
     * split; [reflexivity|apply zero_column_same].
Qed.

Lemma global_hi_rows_safe : forallb (privateRowSafe COL_GLOBAL_VALUE_HI globalGates) FullRelation.rows = true.
Proof. vm_compute. reflexivity. Qed.

Theorem full_global_hi_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> exists w',
 FullRelation.satFull w' /\
 0 <= w' COL_GLOBAL_VALUE_HI < 4294967296 /\
 eventsForRow memories w' = eventsForRow memories w /\
 map (fun '(a,b) => (w' a,w' b)) continuityLinks =
 map (fun '(a,b) => (w a,w b)) continuityLinks /\
 (forall c, c <> COL_GLOBAL_VALUE_HI -> w' c = w c).
Proof.
 intros w Hprime Hfull.
 destruct (Z_lt_ge_dec (w COL_GLOBAL_VALUE_HI) 4294967296) as [Hsmall|Hlarge].
 - exists w. split; [exact Hfull|]. split; [pose proof (value_bound w COL_GLOBAL_VALUE_HI);lia|].
   repeat split; reflexivity.
 - destruct (global_hi_inactive w Hprime Hfull ltac:(lia)) as [Hget Hset].
   exists (zeroColumn COL_GLOBAL_VALUE_HI w). split.
   + eapply private_full_repair with (gates := globalGates).
     * vm_compute; discriminate.
     * exact global_hi_rows_safe.
     * unfold globalGates. repeat constructor; cbn [eval fold_right]; rewrite ?Hget, ?Hset; reflexivity.
     * exact Hfull.
   + split; [rewrite zero_column_value; lia|]. split.
     * cbv beta iota zeta delta [eventsForRow memories flat_map ports memoryId portEvent active activation addressColumns valueColumn portKind map option_map].
       repeat match goal with
       | |- context [value (zeroColumn ?column ?witness) ?c] =>
           let Hneq := fresh in
           assert (Hneq : c <> column) by (vm_compute; discriminate);
           rewrite (zero_column_same column witness c Hneq); clear Hneq
       end.
       normalize_columns. rewrite Hget, Hset. reflexivity.
     * split; [reflexivity|apply zero_column_same].
Qed.

Lemma table_index_rows_safe : forallb (privateRowSafe COL_TABLE_INDEX tableIndexGates) FullRelation.rows = true.
Proof. vm_compute. reflexivity. Qed.

Theorem full_table_index_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> exists w',
 FullRelation.satFull w' /\
 0 <= w' COL_TABLE_INDEX < 4294967296 /\
 eventsForRow memories w' = eventsForRow memories w /\
 map (fun '(a,b) => (w' a,w' b)) continuityLinks =
 map (fun '(a,b) => (w a,w b)) continuityLinks /\
 (forall c, c <> COL_TABLE_INDEX -> w' c = w c).
Proof.
 intros w Hprime Hfull.
 destruct (Z_lt_ge_dec (w COL_TABLE_INDEX) 4294967296) as [Hsmall|Hlarge].
 - exists w. split; [exact Hfull|]. split; [pose proof (value_bound w COL_TABLE_INDEX);lia|].
   repeat split; reflexivity.
 - destruct (table_index_inactive w Hprime Hfull ltac:(lia)) as [Hget [Hset [Hresolve [Hread Hwrite]]]].
   exists (zeroColumn COL_TABLE_INDEX w). split.
   + eapply private_full_repair with (gates := tableIndexGates).
     * vm_compute; discriminate.
     * exact table_index_rows_safe.
     * unfold tableIndexGates. repeat constructor; cbn [eval fold_right]; rewrite ?Hget, ?Hset, ?Hresolve; reflexivity.
     * exact Hfull.
   + split; [rewrite zero_column_value; lia|]. split.
     * cbv beta iota zeta delta [eventsForRow memories flat_map ports memoryId portEvent active activation addressColumns valueColumn portKind map option_map].
       repeat match goal with
       | |- context [value (zeroColumn ?column ?witness) ?c] =>
           let Hneq := fresh in
           assert (Hneq : c <> column) by (vm_compute; discriminate);
           rewrite (zero_column_same column witness c Hneq); clear Hneq
       end.
       normalize_columns. rewrite ?Hget, ?Hset, ?Hresolve, ?Hread, ?Hwrite. reflexivity.
     * split; [reflexivity|apply zero_column_same].
Qed.
