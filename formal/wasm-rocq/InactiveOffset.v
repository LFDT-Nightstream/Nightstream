From Stdlib Require Import List ZArith Znumtheory Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

(* SPEC.md section 7.6: an inactive private column has no exported effect. *)
Lemma sat_nonzero_right : forall (w : Witness) a b,
 fieldPrime -> sat w (a,b,[]) ->
 -modulus < eval w a < modulus -> 0 < eval w b < modulus -> eval w a = 0.
Proof.
 intros w a b Hp Hsat Ha Hb. unfold sat in Hsat. cbn [eval fold_right] in Hsat.
 apply Z.mod_divide in Hsat; [|unfold modulus;lia].
 replace (eval w a * eval w b - 0) with (eval w a * eval w b) in Hsat by ring.
 apply prime_mult in Hsat; [|exact Hp].
 destruct Hsat as [[k Hk]|[k Hk]]; unfold modulus in *; nia.
Qed.

Definition offsetGates : list SparseRow := [[((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1); (GMEM_BYTE, -1); (GMEM_HALF, -1)];
[((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1); (GMEM_LOCAL, -1); (GMEM_OUTPUT, -1)];
[(GMEM_OUTPUT, 1)];
[(GMEM_LOCAL, 1)];
[(COL_SEL_I32_LOAD, 1); (COL_SEL_I32_LOAD8_S, 1); (COL_SEL_I32_LOAD8_U, 1); (COL_SEL_I32_LOAD16_S, 1); (COL_SEL_I32_LOAD16_U, 1); (COL_SEL_I64_LOAD, 1); (COL_SEL_I32_STORE, 1); (COL_SEL_I32_STORE8, 1); (COL_SEL_I32_STORE16, 1); (COL_SEL_I64_STORE, 1); (COL_SEL_I64_STORE8, 1); (COL_SEL_I64_STORE16, 1); (COL_SEL_I64_STORE32, 1); (COL_SEL_I64_LOAD8_U, 1); (COL_SEL_I64_LOAD16_U, 1); (COL_SEL_I64_LOAD32_U, 1); (COL_SEL_I64_LOAD8_S, 1); (COL_SEL_I64_LOAD16_S, 1); (COL_SEL_I64_LOAD32_S, 1); (GMEM_BYTE, 1); (GMEM_HALF, 1)];
[(COL_SEL_I32_LOAD, 1); (COL_SEL_I32_LOAD8_S, 1); (COL_SEL_I32_LOAD8_U, 1); (COL_SEL_I32_LOAD16_S, 1); (COL_SEL_I32_LOAD16_U, 1); (COL_SEL_I64_LOAD, 1); (COL_SEL_I32_STORE, 1); (COL_SEL_I32_STORE8, 1); (COL_SEL_I32_STORE16, 1); (COL_SEL_I64_STORE, 1); (COL_SEL_I64_STORE8, 1); (COL_SEL_I64_STORE16, 1); (COL_SEL_I64_STORE32, 1); (COL_SEL_I64_LOAD8_U, 1); (COL_SEL_I64_LOAD16_U, 1); (COL_SEL_I64_LOAD32_U, 1); (COL_SEL_I64_LOAD8_S, 1); (COL_SEL_I64_LOAD16_S, 1); (COL_SEL_I64_LOAD32_S, 1)]].

Lemma full_offset_bound_or_inactive : forall w,
 fieldPrime -> FullRelation.satFull w ->
 0 <= w COL_LINEAR_MEM_BYTE_OFFSET < 4294967296 \/
 Forall (fun gate => eval w gate = 0) offsetGates.
Proof.
 intros w Hprime Hfull.
 pose proof (value_bound w COL_LINEAR_MEM_BYTE_OFFSET) as Hvalue.
 destruct (Z_lt_ge_dec (w COL_LINEAR_MEM_BYTE_OFFSET) 4294967296) as [Hsmall|Hlarge];
 [left;lia|right].
 circuit_boolean COL_SEL_I32_LOAD B69.
 assert (IB69 : 0 <= w COL_SEL_I32_LOAD <= 1) by lia; clear B69.
 circuit_boolean COL_SEL_I32_LOAD8_S B70.
 assert (IB70 : 0 <= w COL_SEL_I32_LOAD8_S <= 1) by lia; clear B70.
 circuit_boolean COL_SEL_I32_LOAD8_U B71.
 assert (IB71 : 0 <= w COL_SEL_I32_LOAD8_U <= 1) by lia; clear B71.
 circuit_boolean COL_SEL_I32_LOAD16_S B72.
 assert (IB72 : 0 <= w COL_SEL_I32_LOAD16_S <= 1) by lia; clear B72.
 circuit_boolean COL_SEL_I32_LOAD16_U B73.
 assert (IB73 : 0 <= w COL_SEL_I32_LOAD16_U <= 1) by lia; clear B73.
 circuit_boolean COL_SEL_I64_LOAD B74.
 assert (IB74 : 0 <= w COL_SEL_I64_LOAD <= 1) by lia; clear B74.
 circuit_boolean COL_SEL_I32_STORE B75.
 assert (IB75 : 0 <= w COL_SEL_I32_STORE <= 1) by lia; clear B75.
 circuit_boolean COL_SEL_I32_STORE8 B76.
 assert (IB76 : 0 <= w COL_SEL_I32_STORE8 <= 1) by lia; clear B76.
 circuit_boolean COL_SEL_I32_STORE16 B77.
 assert (IB77 : 0 <= w COL_SEL_I32_STORE16 <= 1) by lia; clear B77.
 circuit_boolean COL_SEL_I64_STORE B78.
 assert (IB78 : 0 <= w COL_SEL_I64_STORE <= 1) by lia; clear B78.
 circuit_boolean COL_SEL_I64_STORE8 B137.
 assert (IB137 : 0 <= w COL_SEL_I64_STORE8 <= 1) by lia; clear B137.
 circuit_boolean COL_SEL_I64_STORE16 B138.
 assert (IB138 : 0 <= w COL_SEL_I64_STORE16 <= 1) by lia; clear B138.
 circuit_boolean COL_SEL_I64_STORE32 B139.
 assert (IB139 : 0 <= w COL_SEL_I64_STORE32 <= 1) by lia; clear B139.
 circuit_boolean COL_SEL_I64_LOAD8_U B140.
 assert (IB140 : 0 <= w COL_SEL_I64_LOAD8_U <= 1) by lia; clear B140.
 circuit_boolean COL_SEL_I64_LOAD16_U B141.
 assert (IB141 : 0 <= w COL_SEL_I64_LOAD16_U <= 1) by lia; clear B141.
 circuit_boolean COL_SEL_I64_LOAD32_U B142.
 assert (IB142 : 0 <= w COL_SEL_I64_LOAD32_U <= 1) by lia; clear B142.
 circuit_boolean COL_SEL_I64_LOAD8_S B143.
 assert (IB143 : 0 <= w COL_SEL_I64_LOAD8_S <= 1) by lia; clear B143.
 circuit_boolean COL_SEL_I64_LOAD16_S B144.
 assert (IB144 : 0 <= w COL_SEL_I64_LOAD16_S <= 1) by lia; clear B144.
 circuit_boolean COL_SEL_I64_LOAD32_S B145.
 assert (IB145 : 0 <= w COL_SEL_I64_LOAD32_S <= 1) by lia; clear B145.
 circuit_boolean COL_LINEAR_MEM_OFFSET_IS_1 B224.
 assert (IB224 : 0 <= w COL_LINEAR_MEM_OFFSET_IS_1 <= 1) by lia; clear B224.
 circuit_boolean COL_LINEAR_MEM_OFFSET_IS_2 B225.
 assert (IB225 : 0 <= w COL_LINEAR_MEM_OFFSET_IS_2 <= 1) by lia; clear B225.
 circuit_boolean COL_LINEAR_MEM_OFFSET_IS_3 B226.
 assert (IB226 : 0 <= w COL_LINEAR_MEM_OFFSET_IS_3 <= 1) by lia; clear B226.
 circuit_boolean (List.nth 5 GATHER_KIND 0%nat) B1199.
 assert (IB1199 : 0 <= w (List.nth 5 GATHER_KIND 0%nat) <= 1) by lia; clear B1199.
 circuit_boolean (List.nth 6 GATHER_KIND 0%nat) B1200.
 assert (IB1200 : 0 <= w (List.nth 6 GATHER_KIND 0%nat) <= 1) by lia; clear B1200.
 circuit_boolean GMEM_LOCAL B1210.
 assert (IB1210 : 0 <= w GMEM_LOCAL <= 1) by lia; clear B1210.
 circuit_boolean GMEM_OUTPUT B1211.
 assert (IB1211 : 0 <= w GMEM_OUTPUT <= 1) by lia; clear B1211.
 circuit_boolean GMEM_BYTE B1212.
 assert (IB1212 : 0 <= w GMEM_BYTE <= 1) by lia; clear B1212.
 circuit_boolean GMEM_HALF B1213.
 assert (IB1213 : 0 <= w GMEM_HALF <= 1) by lia; clear B1213.
 circuit_row ([(COL_SEL_I32_LOAD, 1); (COL_SEL_I32_LOAD8_S, 1); (COL_SEL_I32_LOAD8_U, 1); (COL_SEL_I32_LOAD16_S, 1); (COL_SEL_I32_LOAD16_U, 1); (COL_SEL_I64_LOAD, 1); (COL_SEL_I32_STORE, 1); (COL_SEL_I32_STORE8, 1); (COL_SEL_I32_STORE16, 1); (COL_SEL_I64_STORE, 1); (COL_SEL_I64_STORE8, 1); (COL_SEL_I64_STORE16, 1); (COL_SEL_I64_STORE32, 1); (COL_SEL_I64_LOAD8_U, 1); (COL_SEL_I64_LOAD16_U, 1); (COL_SEL_I64_LOAD32_U, 1); (COL_SEL_I64_LOAD8_S, 1); (COL_SEL_I64_LOAD16_S, 1); (COL_SEL_I64_LOAD32_S, 1); (GMEM_BYTE, 1); (GMEM_HALF, 1)], [(COL_LINEAR_MEM_BYTE_OFFSET, 1); (COL_LINEAR_MEM_OFFSET_IS_1, -1); (COL_LINEAR_MEM_OFFSET_IS_2, -2); (COL_LINEAR_MEM_OFFSET_IS_3, -3)], ([] : SparseRow)) HNorm.
 circuit_row ([((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1); (GMEM_BYTE, -1); (GMEM_HALF, -1)], [(COL_LINEAR_MEM_BYTE_OFFSET, 1)], ([] : SparseRow)) HZero.
 circuit_row ([((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1)], [(COL_HOST_EVENT_SLOT_VARIANT, 1)], [(GMEM_LOCAL, 1); (GMEM_BYTE, 2); (GMEM_HALF, 4); (GMEM_OUTPUT, 8)]) HDesc.
 pose proof (sat_nonzero_right w [(COL_SEL_I32_LOAD, 1); (COL_SEL_I32_LOAD8_S, 1); (COL_SEL_I32_LOAD8_U, 1); (COL_SEL_I32_LOAD16_S, 1); (COL_SEL_I32_LOAD16_U, 1); (COL_SEL_I64_LOAD, 1); (COL_SEL_I32_STORE, 1); (COL_SEL_I32_STORE8, 1); (COL_SEL_I32_STORE16, 1); (COL_SEL_I64_STORE, 1); (COL_SEL_I64_STORE8, 1); (COL_SEL_I64_STORE16, 1); (COL_SEL_I64_STORE32, 1); (COL_SEL_I64_LOAD8_U, 1); (COL_SEL_I64_LOAD16_U, 1); (COL_SEL_I64_LOAD32_U, 1); (COL_SEL_I64_LOAD8_S, 1); (COL_SEL_I64_LOAD16_S, 1); (COL_SEL_I64_LOAD32_S, 1); (GMEM_BYTE, 1); (GMEM_HALF, 1)] [(COL_LINEAR_MEM_BYTE_OFFSET, 1); (COL_LINEAR_MEM_OFFSET_IS_1, -1); (COL_LINEAR_MEM_OFFSET_IS_2, -2); (COL_LINEAR_MEM_OFFSET_IS_3, -3)] Hprime HNorm) as GNorm.
 cbn [eval fold_right] in GNorm. normalize_columns.
 assert (HNormGate : eval w [(COL_SEL_I32_LOAD, 1); (COL_SEL_I32_LOAD8_S, 1); (COL_SEL_I32_LOAD8_U, 1); (COL_SEL_I32_LOAD16_S, 1); (COL_SEL_I32_LOAD16_U, 1); (COL_SEL_I64_LOAD, 1); (COL_SEL_I32_STORE, 1); (COL_SEL_I32_STORE8, 1); (COL_SEL_I32_STORE16, 1); (COL_SEL_I64_STORE, 1); (COL_SEL_I64_STORE8, 1); (COL_SEL_I64_STORE16, 1); (COL_SEL_I64_STORE32, 1); (COL_SEL_I64_LOAD8_U, 1); (COL_SEL_I64_LOAD16_U, 1); (COL_SEL_I64_LOAD32_U, 1); (COL_SEL_I64_LOAD8_S, 1); (COL_SEL_I64_LOAD16_S, 1); (COL_SEL_I64_LOAD32_S, 1); (GMEM_BYTE, 1); (GMEM_HALF, 1)] = 0).
 { apply GNorm; unfold modulus in *; lia. }
 cbn [eval fold_right] in HNormGate. normalize_columns.
 pose proof (sat_nonzero_right w [((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1); (GMEM_BYTE, -1); (GMEM_HALF, -1)] [(COL_LINEAR_MEM_BYTE_OFFSET, 1)] Hprime HZero) as GZero.
 cbn [eval fold_right] in GZero. normalize_columns.
 assert (HZeroGate : eval w [((List.nth 5 GATHER_KIND 0%nat), 1); ((List.nth 6 GATHER_KIND 0%nat), 1); (GMEM_BYTE, -1); (GMEM_HALF, -1)] = 0).
 { apply GZero; unfold modulus in *; lia. }
 cbn [eval fold_right] in HZeroGate. normalize_columns.
 assert (Hr : w (List.nth 5 GATHER_KIND 0%nat) = 0) by (normalize_columns; lia).
 assert (Hw : w (List.nth 6 GATHER_KIND 0%nat) = 0) by (normalize_columns; lia).
 expand_row HDesc. rewrite Hr, Hw in HDesc.
 apply Z.mod_divide in HDesc; [|unfold modulus;lia].
 destruct HDesc as [k HDesc].
 apply Forall_forall. intros gate Hin.
 unfold offsetGates in Hin. cbn in Hin.
 repeat destruct Hin as [<- | Hin]; try contradiction;
 cbn [eval fold_right]; normalize_columns; unfold modulus in *; nia.
Qed.

Definition omitsColumn (column : nat) (terms : SparseRow) : bool :=
 forallb (fun term => negb (Nat.eqb (fst term) column)) terms.
Definition offsetRowSafe (row : Row) : bool :=
 let '(a,b,c) := row in
 omitsColumn COL_LINEAR_MEM_BYTE_OFFSET a && omitsColumn COL_LINEAR_MEM_BYTE_OFFSET c &&
 (omitsColumn COL_LINEAR_MEM_BYTE_OFFSET b ||
  (termsEqual c [] && existsb (termsEqual a) offsetGates)).

Lemma eval_omitted_column : forall (w w' : Witness) column terms,
 (forall c, c <> column -> w' c = w c) -> omitsColumn column terms = true ->
 eval w' terms = eval w terms.
Proof.
 intros w w' column terms Hagree Hsafe.
 induction terms as [|[c k] rest IH]; [reflexivity|].
 cbn [omitsColumn forallb fst] in Hsafe.
 apply andb_true_iff in Hsafe as [Hc Hr].
 apply negb_true_iff in Hc. apply Nat.eqb_neq in Hc.
 change (k * w' c + eval w' rest = k * w c + eval w rest).
 rewrite Hagree by exact Hc. rewrite IH by exact Hr. reflexivity.
Qed.

Lemma offset_row_repair : forall (w w' : Witness) row,
 (forall c, c <> COL_LINEAR_MEM_BYTE_OFFSET -> w' c = w c) ->
 Forall (fun gate => eval w gate = 0) offsetGates ->
 offsetRowSafe row = true -> sat w row -> sat w' row.
Proof.
 intros w w' [[a b] c] Hagree Hgates Hsafe Hsat.
 unfold offsetRowSafe in Hsafe.
 apply andb_true_iff in Hsafe as [Hac Hb].
 apply andb_true_iff in Hac as [Ha Hc].
 unfold sat in *. rewrite (eval_omitted_column w w' _ a Hagree Ha).
 rewrite (eval_omitted_column w w' _ c Hagree Hc).
 apply orb_true_iff in Hb as [Hb|Hb].
 - rewrite (eval_omitted_column w w' _ b Hagree Hb). exact Hsat.
 - apply andb_true_iff in Hb as [Hempty Hmember].
   apply terms_equal_sound in Hempty. subst c.
   apply existsb_exists in Hmember as [gate [Hin Heq]].
   apply terms_equal_sound in Heq. subst gate.
   pose proof (proj1 (Forall_forall _ _) Hgates a Hin) as Hz.
   rewrite Hz. cbn [eval fold_right]. reflexivity.
Qed.

Definition zeroOffset (w : Witness) : Witness :=
 {| entries := fun c => if Nat.eq_dec c COL_LINEAR_MEM_BYTE_OFFSET
    then exist (fun z : Z => 0 <= z < modulus) 0 ltac:(unfold modulus; lia) else entries w c |}.
Lemma zero_offset_same : forall w c, c <> COL_LINEAR_MEM_BYTE_OFFSET -> zeroOffset w c = w c.
Proof.
 intros w c Hc. unfold value, zeroOffset. cbn.
 destruct (Nat.eq_dec c COL_LINEAR_MEM_BYTE_OFFSET); [contradiction|reflexivity].
Qed.
Lemma zero_offset_value : forall w, zeroOffset w COL_LINEAR_MEM_BYTE_OFFSET = 0.
Proof.
 intro w. unfold value, zeroOffset. cbn.
 destruct (Nat.eq_dec COL_LINEAR_MEM_BYTE_OFFSET COL_LINEAR_MEM_BYTE_OFFSET); congruence.
Qed.

Lemma zero_offset_memory_events : forall w,
 eventsForRow memories (zeroOffset w) = eventsForRow memories w.
Proof. intro w. reflexivity. Qed.

Lemma zero_offset_continuity : forall w,
 map (fun '(a,b) => (zeroOffset w a, zeroOffset w b)) continuityLinks =
 map (fun '(a,b) => (w a, w b)) continuityLinks.
Proof. intro w. reflexivity. Qed.

Lemma offset_rows_safe : forallb offsetRowSafe FullRelation.rows = true.
Proof. vm_compute. reflexivity. Qed.

Theorem full_offset_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> exists w',
 FullRelation.satFull w' /\
 0 <= w' COL_LINEAR_MEM_BYTE_OFFSET < 4294967296 /\
 eventsForRow memories w' = eventsForRow memories w /\
 map (fun '(a,b) => (w' a,w' b)) continuityLinks =
 map (fun '(a,b) => (w a,w b)) continuityLinks /\
 (forall c, c <> COL_LINEAR_MEM_BYTE_OFFSET -> w' c = w c).
Proof.
 intros w Hprime Hfull.
 destruct (full_offset_bound_or_inactive w Hprime Hfull) as [Hbound|Hgates].
 - exists w. split; [exact Hfull|]. split; [exact Hbound|].
   repeat split; reflexivity.
 - exists (zeroOffset w). split.
   + split.
     * change (zeroOffset w COL_ONE = 1).
       rewrite zero_offset_same by (vm_compute; discriminate). exact (proj1 Hfull).
     * apply Forall_forall. intros row Hin.
       eapply offset_row_repair; [exact (zero_offset_same w)|exact Hgates| |].
       -- exact (proj1 (forallb_forall _ _) offset_rows_safe row Hin).
       -- exact (proj1 (Forall_forall _ _) (proj2 Hfull) row Hin).
   + split; [rewrite zero_offset_value; lia|].
     split; [apply zero_offset_memory_events|].
     split; [apply zero_offset_continuity|apply zero_offset_same].
Qed.
