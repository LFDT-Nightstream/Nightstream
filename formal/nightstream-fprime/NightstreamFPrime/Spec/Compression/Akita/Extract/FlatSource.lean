import NightstreamFPrime.Spec.Compression.Akita.FlatFold

/-! The checks of one packing fold of Akita for a flat witness that the next level commits to
(ePrint 2026/1983, revision of 2026-10-15, Section 10.3, "Successor source": the decoder of the
schedule maps a source `σ` to a table `w_σ` in the witness order of (76), "on which the exact
branch relation can be evaluated", and Lemma 10.42 (i)).

- `OpeningShape.ofFlat` reads the digit vectors of one fold from a table in the order of
  `Witness.flat`, and `OpeningShape.flat_ofFlat` states that their flat witness is the table on
  the cells before `OpeningShape.cells`.
- A committed table can have each value in the cells from `OpeningShape.cells`. The rows of
  `flatRelation` read only cells before `OpeningShape.cells`: a zero of a padded vector is a
  coefficient 0 of the row, as in Akita (109).
- `satisfies_of_table` gives the exact branch relation of the digit vectors `ofFlat` of a table
  with `flatRelation`, the digit range statement and the binary statement.
- `tableFold_satisfies` is `flatFold_satisfies` for a table: on a path of `foldCheck` that avoids
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

/-- Two tables with the same cells before `bound` give one value to a row that reads cells
before `bound`. -/
theorem RingRow.holds_congr {row : RingRow} {bound : ℕ} (below : row.Below bound)
    {table other : Table} (same : ∀ x < bound, table x = other x) (holds : row.Holds table) :
    row.Holds other := by
  have each : ∀ c j, (row.address c j).elim 0 table = (row.address c j).elim 0 other :=
    fun c j => by
      cases read : row.address c j with
      | none => rfl
      | some x => exact same x (below c j x read)
  unfold RingRow.Holds RingRow.residual RingRow.read at holds ⊢
  simpa only [each] using holds

variable (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)

/-- Two tables with the same cells before `cells` satisfy `flatRelation` together. -/
theorem flatRelation_holds_congr {table other : Table}
    (same : ∀ x < O.cells fold, table x = other x)
    (holds : (flatRelation O fold β st challenge μ fits).Holds table) :
    (flatRelation O fold β st challenge μ fits).Holds other := by
  refine ⟨fun r => RingRow.holds_congr (flatRows_below O fold β st challenge r) same (holds.1 r),
    ?_⟩
  rw [← holds.2]
  refine Finset.sum_congr rfl fun u _ => ?_
  show algebraMap Zq E (other u) * scalarWeight O fold β st.point u =
    algebraMap Zq E (table u) * scalarWeight O fold β st.point u
  by_cases inside : u < O.cells fold
  · rw [same u inside]
  · rw [scalarWeight_of_le O fold β st.point
      ((O.chainStart_le_cells fold).trans (Nat.le_of_not_lt inside)), mul_zero, mul_zero]

variable {β st challenge μ fits}

/-- A table with `flatRelation`, the digit range statement on the cells before `cells` and the
binary statement on `I_bin` has digit vectors `ofFlat` with the exact branch relation
`Witness.Satisfies`. `zero`: the value 0 is in the range of `Q_sq`. The table can have each value
from the cell `cells`. -/
theorem satisfies_of_table (small : S.certifiedBase < fieldModulus) {table : Table}
    (relation : (flatRelation O fold β st challenge μ fits).Holds table)
    (range : ∀ x < O.cells fold, rangeSq S.certifiedBase (table x * (table x + 1)) = 0)
    (zero : rangeSq S.certifiedBase 0 = 0)
    (binary : ∀ x, O.binarySupport fold x → table x * (table x + 1) = 0) :
    (O.ofFlat fold table).Satisfies β st challenge := by
  have same : ∀ x < O.cells fold, table x = (O.ofFlat fold table).flat x :=
    fun x inside => (O.flat_ofFlat fold table inside).symm
  have ranged : RangeHolds S.certifiedBase μ (O.ofFlat fold table).flat := fun x _ => by
    by_cases below : x < O.cells fold
    · rw [← same x below]
      exact range x below
    · rw [Witness.flat_of_le _ (Nat.le_of_not_lt below), zero_mul]
      exact zero
  have binaried : BinaryOn (O.binarySupport fold) μ (O.ofFlat fold table).flat :=
    fun x _ member => by
      rw [← same x member.2]
      exact binary x member
  exact Witness.satisfies_of_flat μ fits
    ((O.flatLayout fold μ fits).alphabets small _ ranged binaried)
    (flatRelation_holds_congr β st challenge μ fits same relation)

variable [Fintype E] [DecidableEq E] (β st challenge μ fits)

/-- Akita Lemma 10.42 (i) for the table of a successor source, with `foldCheck_avoids`: on a path
of `foldCheck` that avoids the bad sets, a final claim that is true for the table gives the exact
branch relation `Witness.Satisfies` of the digit vectors `ofFlat` of the table (the hypothesis of
Akita Corollary 10.32 for one child). -/
theorem tableFold_satisfies (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold table).Satisfies β st challenge :=
  (foldCheck_avoids (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
    (O.binarySupport fold) (O.compressed fold) (O.binarySupport_empty fold) table).mono fun _
      checked r2 v equal opening =>
      satisfies_of_table small (checked r2 v equal opening).1
        (fun x inside => (checked r2 v equal opening).2.1 x (inside.trans_le fits))
        (checked r2 v equal opening).2.1.zero
        (fun x member => (checked r2 v equal opening).2.2 x (member.2.trans_le fits) member)

end Rows

end NightstreamFPrime.Spec.Compression.Akita
