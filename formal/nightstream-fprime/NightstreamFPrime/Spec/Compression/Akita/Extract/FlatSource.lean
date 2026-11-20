import NightstreamFPrime.Spec.Compression.Akita.FlatFold

/-! The checks of one packing fold of Akita for a flat witness that the next level commits to
(ePrint 2026/1983, revision of 2026-10-15, Section 10.3, "Successor source": the decoder of the
schedule maps a source `σ` to a table `w_σ` in the witness order of (76), "on which the exact
branch relation can be evaluated", and Lemma 10.42 (i)).

- `OpeningShape.ofFlat` reads the digit vectors of one fold from a table in the order of
  `Witness.flat`, and `OpeningShape.flat_ofFlat` states that their flat witness is the table on
  the cells before `OpeningShape.cells`.
- A committed table can have each value in the cells from `OpeningShape.cells`. The rows of
  `flatRelation` read the cell `OpeningShape.cells` for each zero of a padded vector, so
  `lockedRelation` adds the ring row `w_cells = 0`. Akita has no such row, because its rows do
  not read a zero from the witness.
- `lockedReduction` is the quotient-free reduction of `lockedRelation`, and
  `lockedFold_satisfies` is `flatFold_satisfies` for a table: on a path of `foldCheck` that avoids
  the bad sets, a true final claim for the table gives the exact branch relation of the digit
  vectors `ofFlat` of the table. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Tables

variable {d m : ℕ} [NeZero d]

/-- The sequence of the vector that reads `m` elements of `R_{q,d}` from a table is the table on
its first `m·d` cells. -/
theorem sequence_ofCoeffs (table : Table) {x : ℕ} (small : x < m * d) :
    sequence (fun j : Fin m => (ofCoeffs fun k => table (j * d + k) : Rq d)) x = table x := by
  have positive := NeZero.pos d
  have inside : x / d < m := (Nat.div_lt_iff_lt_mul positive).mpr small
  have read : sequence (fun j : Fin m => (ofCoeffs fun k => table (j * d + k) : Rq d))
      (x / d * d + x % d) = coeff (ofCoeffs fun k => table (x / d * d + k) : Rq d) (x % d) :=
    sequence_apply _ ⟨x / d, inside⟩ (Nat.mod_lt x positive)
  rw [Nat.div_add_mod'] at read
  rw [read, coeff_ofCoeffs _ (Nat.mod_lt x positive), Nat.div_add_mod']

/-- The vector whose coefficients are the cells of a table in the order of `code`: the inverse of
`cellsOf`. -/
noncomputable def vectorOf {ι : Type} (code : ι ≃ Fin m) (table : Table) : ι → Rq d :=
  fun j => ofCoeffs fun k => table (code j * d + k)

theorem cellsOf_vectorOf {ι : Type} (code : ι ≃ Fin m) (table : Table) {x : ℕ}
    (small : x < m * d) : cellsOf code (vectorOf code table : ι → Rq d) x = table x := by
  unfold cellsOf vectorOf
  simp only [Equiv.apply_symm_apply]
  exact sequence_ofCoeffs table small

/-- The digit vectors of a compression path whose sequences are two tables. -/
noncomputable def PathDigits.ofTables :
    (chain : Option ChainShape) → Table → Table → PathDigits chain
  | none, _, _ => PUnit.unit
  | some c, first, second =>
      (⟨fun j => ofCoeffs fun k => first (j * c.firstDegree + k),
        fun j => ofCoeffs fun k => second (j * c.secondDegree + k)⟩ : ChainDigits c)

theorem PathDigits.firstTable_ofTables : (chain : Option ChainShape) → (first second : Table) →
    (x : ℕ) → x < firstCells chain →
    PathDigits.firstTable chain (PathDigits.ofTables chain first second) x = first x
  | none, _, _, _, small => absurd small (Nat.not_lt_zero _)
  | some _, first, _, _, small => sequence_ofCoeffs first small

theorem PathDigits.secondTable_ofTables : (chain : Option ChainShape) → (first second : Table) →
    (x : ℕ) → x < secondCells chain →
    PathDigits.secondTable chain (PathDigits.ofTables chain first second) x = second x
  | none, _, _, _, small => absurd small (Nat.not_lt_zero _)
  | some _, _, second, _, small => sequence_ofCoeffs second small

end Tables

namespace OpeningShape

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- The digit vectors of one fold that a table has in the order of `Witness.flat` (Akita (75)
and (76)): the decoder of a successor source to the witness of the exact branch relation. -/
noncomputable def ofFlat (table : Table) : Witness O fold where
  inner i entry := ofCoeffs fun c =>
    if inside : c / S.outerDegree < S.ratio then
      coeff ((vectorOf S.innerCode fun x => table (O.innerStart fold + x) :
        Fin S.blocks × S.OuterColumn → Rq S.outerDegree)
          (i, (entry.1, ⟨c / S.outerDegree, inside⟩, entry.2))) (c % S.outerDegree)
    else 0
  chain := PathDigits.ofTables S.chain (fun x => table (O.chainStart fold + x))
    fun x => table (O.secondStart fold + x)
  partials := vectorOf O.partialCode fun x => table (S.responseCells fold + x)
  openChain := PathDigits.ofTables O.chain (fun x => table (O.openFirstStart fold + x))
    fun x => table (O.openSecondStart fold + x)
  response := vectorOf (S.responseCode fold) table

theorem outerView_ofFlat (table : Table) (i : Fin S.blocks) (column : S.OuterColumn) :
    S.outerView ((O.ofFlat fold table).inner i) column =
      vectorOf S.innerCode (fun x => table (O.innerStart fold + x)) (i, column) := by
  obtain ⟨a, y, u⟩ := column
  refine ofCoeffs_eq fun k small => ?_
  have address := S.address_lt y small
  have quotient : (y * S.outerDegree + k) / S.outerDegree = y := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (NeZero.pos _), Nat.div_eq_of_lt small, zero_add]
  have rest : (y * S.outerDegree + k) % S.outerDegree = k := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]
  show coeff (ofCoeffs fun c =>
    if inside : c / S.outerDegree < S.ratio then
      coeff ((vectorOf S.innerCode fun x => table (O.innerStart fold + x) :
        Fin S.blocks × S.OuterColumn → Rq S.outerDegree)
          (i, (a, ⟨c / S.outerDegree, inside⟩, u))) (c % S.outerDegree)
    else 0 : Rq S.innerDegree) (y * S.outerDegree + k) = _
  rw [coeff_ofCoeffs _ address]
  simp only [quotient, rest, Fin.is_lt, ↓reduceDIte, Fin.eta]

/-- The flat witness of the digit vectors of a table is the table on the cells before
`cells`. -/
theorem flat_ofFlat (table : Table) {x : ℕ} (small : x < O.cells fold) :
    (O.ofFlat fold table).flat x = table x := by
  have views : (fun entry : Fin S.blocks × S.OuterColumn =>
      S.outerView ((O.ofFlat fold table).inner entry.1) entry.2) =
        vectorOf S.innerCode fun x => table (O.innerStart fold + x) :=
    funext fun entry => O.outerView_ofFlat fold table entry.1 entry.2
  unfold cells openSecondStart secondStart openFirstStart chainStart innerStart at small
  by_cases response : x < S.responseCells fold
  · rw [Witness.flat_response _ response]
    exact cellsOf_vectorOf (S.responseCode fold) table response
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt response)
  by_cases partials : x < O.partialCells
  · rw [Witness.flat_partial _ partials]
    exact cellsOf_vectorOf O.partialCode (fun x => table (S.responseCells fold + x)) partials
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt partials)
  rw [← Nat.add_assoc, ← show O.innerStart fold = S.responseCells fold + O.partialCells from rfl]
  by_cases inner : x < S.innerCells
  · rw [Witness.flat_inner _ inner, views]
    exact cellsOf_vectorOf S.innerCode (fun x => table (O.innerStart fold + x)) inner
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt inner)
  rw [← Nat.add_assoc, ← show O.chainStart fold = O.innerStart fold + S.innerCells from rfl]
  by_cases first : x < firstCells S.chain
  · rw [Witness.flat_first _ first]
    exact PathDigits.firstTable_ofTables S.chain (fun x => table (O.chainStart fold + x))
      (fun x => table (O.secondStart fold + x)) x first
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt first)
  rw [← Nat.add_assoc,
    ← show O.openFirstStart fold = O.chainStart fold + firstCells S.chain from rfl]
  by_cases openFirst : x < firstCells O.chain
  · rw [Witness.flat_openFirst _ openFirst]
    exact PathDigits.firstTable_ofTables O.chain (fun x => table (O.openFirstStart fold + x))
      (fun x => table (O.openSecondStart fold + x)) x openFirst
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt openFirst)
  rw [← Nat.add_assoc,
    ← show O.secondStart fold = O.openFirstStart fold + firstCells O.chain from rfl]
  by_cases second : x < secondCells S.chain
  · rw [Witness.flat_second _ second]
    exact PathDigits.secondTable_ofTables S.chain (fun x => table (O.chainStart fold + x))
      (fun x => table (O.secondStart fold + x)) x second
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt second)
  rw [← Nat.add_assoc,
    ← show O.openSecondStart fold = O.secondStart fold + secondCells S.chain from rfl]
  rw [Witness.flat_openSecond _ (show x < secondCells O.chain by omega)]
  exact PathDigits.secondTable_ofTables O.chain (fun x => table (O.openFirstStart fold + x))
    (fun x => table (O.openSecondStart fold + x)) x (by omega)

end OpeningShape

section Rows

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Algebra Zq E]

/-- Two tables with the same cells up to `bound` give one value to a row that reads cells up to
`bound`. -/
theorem RingRow.holds_congr {row : RingRow} {bound : ℕ} (below : row.Below bound)
    {table other : Table} (same : ∀ x ≤ bound, table x = other x) (holds : row.Holds table) :
    row.Holds other := by
  have each : ∀ c j, table (row.address c j) = other (row.address c j) :=
    fun c j => same _ (below c j)
  unfold RingRow.Holds RingRow.residual RingRow.read at holds ⊢
  simpa only [each] using holds

variable (O) (fold)

/-- The left side of the ring row `w_cells = 0`: one occurrence with the multiplier 1 that reads
the cell `cells` at each coefficient. -/
noncomputable def zeroForm : Form O.carrierDegree where
  Occurrence := Unit
  multiplier _ := 1
  address _ _ := O.cells fold

theorem zero_of_zeroForm {table : Table} (holds : ((zeroForm O fold).row 0).Holds table) :
    table (O.cells fold) = 0 := by
  have value := ((zeroForm O fold).row_holds 0 table).mp holds
  have read : (zeroForm O fold).value table = ofCoeffs fun _ => table (O.cells fold) := by
    show (∑ _c : Unit, (1 : Rq O.carrierDegree) * ofCoeffs fun _ => table (O.cells fold)) = _
    simp
  rw [read] at value
  have zero := congrArg (fun x => coeff x 0) value
  simpa [coeff_ofCoeffs _ (NeZero.pos O.carrierDegree)] using zero

/-- The weight of the scalar row is 0 at each cell after `cells`. -/
theorem scalarWeight_of_lt (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E) {x : ℕ}
    (large : O.cells fold < x) : scalarWeight O fold β point x = 0 :=
  Finset.sum_eq_zero fun entry _ => ite_eq_right fun same => by
    have below := O.partialPosition_le fold entry.1 entry.2.2.2
      (entry.2.2.1 * O.carrierDegree + entry.2.1)
    omega

variable (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold < 2 ^ μ)

/-- The ring rows of `flatRows`, and the row `w_cells = 0`. -/
noncomputable def lockedRows : Fin (O.rowCount + 1) → RingRow :=
  Fin.append (flatRows O fold β st challenge) fun _ : Fin 1 => (zeroForm O fold).row 0

theorem lockedRows_below (r : Fin (O.rowCount + 1)) :
    (lockedRows O fold β st challenge r).Below (O.cells fold) := by
  revert r
  exact (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr
    ⟨flatRows_below O fold β st challenge,
      fun _ => Form.Below.row (f := zeroForm O fold) (fun _ _ => le_rfl) 0⟩

theorem lockedRows_degree_le (r : Fin (O.rowCount + 1)) :
    (lockedRows O fold β st challenge r).degree ≤ O.largestDegree := by
  revert r
  exact (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
    ⟨flatRows_degree_le O fold β st challenge,
      fun _ => le_max_of_le_left (le_max_left O.carrierDegree S.innerDegree)⟩

/-- The relation of one fold for a table that the next level commits to: the rows and the scalar
row of `flatRelation`, and the row `w_cells = 0`. -/
noncomputable def lockedRelation : FlatRelation E where
  vars := μ
  ring := O.rowCount + 1
  row := lockedRows O fold β st challenge
  inside r c j _ := (lockedRows_below O fold β st challenge r c j).trans_lt fits
  weight := scalarWeight O fold β st.point
  value := st.value

variable {O fold β st challenge μ fits}

theorem lockedRelation_holds {table : Table}
    (holds : (lockedRelation O fold β st challenge μ fits).Holds table) :
    table (O.cells fold) = 0 ∧ (flatRelation O fold β st challenge μ fits).Holds table := by
  have rows : ∀ r, (Fin.append (flatRows O fold β st challenge)
      (fun _ : Fin 1 => (zeroForm O fold).row 0) r).Holds table := holds.1
  obtain ⟨flat, zero⟩ := (forall_append (RingRow.Holds · table)
    (flatRows O fold β st challenge) (fun _ : Fin 1 => (zeroForm O fold).row 0)).mp rows
  exact ⟨zero_of_zeroForm O fold (zero 0), flat, holds.2⟩

/-- Two tables with the same cells up to `cells` satisfy `flatRelation` together. -/
theorem flatRelation_holds_congr {table other : Table}
    (same : ∀ x ≤ O.cells fold, table x = other x)
    (holds : (flatRelation O fold β st challenge μ fits).Holds table) :
    (flatRelation O fold β st challenge μ fits).Holds other := by
  refine ⟨fun r => RingRow.holds_congr (flatRows_below O fold β st challenge r) same (holds.1 r),
    ?_⟩
  rw [← holds.2]
  refine Finset.sum_congr rfl fun u _ => ?_
  show algebraMap Zq E (other u) * scalarWeight O fold β st.point u =
    algebraMap Zq E (table u) * scalarWeight O fold β st.point u
  by_cases inside : u ≤ O.cells fold
  · rw [same u inside]
  · rw [scalarWeight_of_lt O fold β st.point (Nat.lt_of_not_le inside), mul_zero, mul_zero]

variable [Fintype E] (β st challenge μ fits)

/-- The quotient-free reduction of `lockedRelation` (Akita Section 7.3 and Theorem 7.4), with
`2^batch` places for the rows. Its bad set has at most `d_max − 1` points. -/
noncomputable def lockedReduction (batch : ℕ) (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) :
    Reduction E :=
  (lockedRelation O fold β st challenge μ fits).free (algebraMap Zq E).injective O.largestDegree
    batch (lockedRows_degree_le O fold β st challenge) rowsFit

variable [DecidableEq E]

/-- Akita Lemma 10.42 (i) for the table of a successor source, with `foldCheck_avoids`: on a path
of `foldCheck` that avoids the bad sets, a final claim that is true for the table gives the exact
branch relation `Witness.Satisfies` of the digit vectors `ofFlat` of the table (the hypothesis of
Akita Corollary 10.32 for one child). -/
theorem lockedFold_satisfies (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (lockedReduction β st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold table).Satisfies β st challenge := by
  refine (foldCheck_avoids (lockedReduction β st challenge μ fits batch rowsFit) S.certifiedBase
    (O.binarySupport fold) table).mono fun result checked r2 v equal opening => ?_
  obtain ⟨relation, range, binary⟩ := checked r2 v equal opening
  have range : RangeHolds S.certifiedBase μ table := range
  have binary : BinaryOn (O.binarySupport fold) μ table := binary
  obtain ⟨zero, flat⟩ := lockedRelation_holds relation
  have same : ∀ x ≤ O.cells fold, table x = (O.ofFlat fold table).flat x := fun x below => by
    rcases Nat.lt_or_eq_of_le below with inside | rfl
    · exact (O.flat_ofFlat fold table inside).symm
    · rw [zero, Witness.flat_of_le _ le_rfl]
  have ranged : RangeHolds S.certifiedBase μ (O.ofFlat fold table).flat := fun x inside => by
    by_cases below : x ≤ O.cells fold
    · rw [← same x below]
      exact range x inside
    · have last := range (O.cells fold) fits
      rw [zero] at last
      rw [Witness.flat_of_le _ (Nat.le_of_lt (Nat.lt_of_not_le below))]
      exact last
  have binaried : BinaryOn (O.binarySupport fold) μ (O.ofFlat fold table).flat :=
    fun x inside member => by
      rw [← same x (Nat.le_of_lt member.2)]
      exact binary x inside member
  exact Witness.satisfies_of_flat μ fits
    ((O.flatLayout fold μ fits).alphabets small _ ranged binaried)
    (flatRelation_holds_congr same flat)

end Rows

end NightstreamFPrime.Spec.Compression.Akita
