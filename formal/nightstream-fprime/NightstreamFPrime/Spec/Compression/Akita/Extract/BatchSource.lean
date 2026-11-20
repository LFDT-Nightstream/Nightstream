import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchRows
import NightstreamFPrime.Spec.Compression.Akita.Extract.Honest

/-! The checks of one fold of Akita with opening groups for a flat witness that the next level
commits to, and their completeness (ePrint 2026/1983, revision of 2026-10-15, Section 10.3,
"Successor source", Lemma 10.42 (i), and Lemma 10.7 with Theorem 10.9).

- `Batch.ofFlat` reads the digit vectors of the groups from a table in the order of
  `Batch.Witness.flat`, and `Batch.flat_ofFlat` states that their flat witness is the table on
  the cells before `Batch.cells`.
- `Batch.lockedRelation` adds the ring row `w_cells = 0` to `Batch.flatRelation`, as
  `lockedRelation` for one group. `Batch.lockedReduction` is its quotient-free reduction.
- `Batch.satisfies_of_locked` gives `Batch.Witness.Satisfies` of the digit vectors `Batch.ofFlat`
  of a table with `Batch.lockedRelation`, the digit range statement and the binary statement.
- `Batch.lockedFold_satisfies` is `lockedFold_satisfies` for the groups: on a path of `foldCheck`
  that avoids the bad sets, a true final claim for the table gives `Batch.Witness.Satisfies` of
  the digit vectors `Batch.ofFlat` of the table.
- `Batch.Witness.Alphabets.rangeHolds` and `Batch.Witness.Alphabets.binaryOn` give the digit
  range statement and the binary statement of the flat witness from the alphabets, and
  `Batch.flatFold_winsTo` and `Batch.lockedFold_winsTo` are the completeness of the checks. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- Two pairs of tables with the same cells in the two digit sequences give the same digit
vectors of a compression path. -/
theorem PathDigits.ofTables_congr : (chain : Option ChainShape) →
    {first second first' second' : Table} → (∀ x < firstCells chain, first x = first' x) →
    (∀ x < secondCells chain, second x = second' x) →
    PathDigits.ofTables chain first second = PathDigits.ofTables chain first' second'
  | none, _, _, _, _, _, _ => rfl
  | some c, first, second, first', second', firsts, seconds => by
      show (PathDigits.ofTables (some c) first second : ChainDigits c) =
        PathDigits.ofTables (some c) first' second'
      exact ChainDigits.ext
        (funext fun j => ofCoeffs_eq fun k small =>
          (firsts _ (cell_lt j small)).trans
            (coeff_ofCoeffs (fun k => first' (j * c.firstDegree + k)) small).symm)
        (funext fun j => ofCoeffs_eq fun k small =>
          (seconds _ (cell_lt j small)).trans
            (coeff_ofCoeffs (fun k => second' (j * c.secondDegree + k)) small).symm)

namespace Batch

variable {n : ℕ} (B : Batch (Fin n))

/-- The digit vectors of one fold with opening groups that a table has in the order of
`Batch.Witness.flat` (Akita (75) and (76)): the decoder of a successor source to the witness of
the exact branch relation. The digit vectors of the group `g` are those of `OpeningShape.ofFlat`
for the cells of the group. -/
noncomputable def ofFlat (table : Table) : B.Witness where
  inner g := ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).inner
  chain g := ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).chain
  partials column :=
    ((B.opening column.1).ofFlat (B.fold column.1) fun x =>
      table (B.relocate column.1 x)).partials column.2
  openChain := PathDigits.ofTables B.chain (fun x => table (B.openFirstStart + x))
    fun x => table (B.openSecondStart + x)
  response g := ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).response

theorem ofFlat_group (table : Table) (g : Fin n) :
    (B.ofFlat table).group g = (B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x) := by
  have chain : (B.ofFlat table).openChain =
      ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).openChain :=
    PathDigits.ofTables_congr B.chain
      (fun x small => congrArg table (B.relocate_openFirst g small).symm)
      (fun x _ => congrArg table (B.relocate_openSecond g x).symm)
  exact congrArg (fun digits =>
    (⟨⟨((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).inner,
      ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).chain,
      ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).partials, digits⟩,
      ((B.opening g).ofFlat (B.fold g) fun x => table (B.relocate g x)).response⟩ :
        Akita.Witness (B.opening g) (B.fold g))) chain

/-- The flat witness of the digit vectors of a table is the table on the cells before
`cells`. -/
theorem flat_ofFlat (table : Table) {x : ℕ} (small : x < B.cells) :
    (B.ofFlat table).flat x = table x := by
  have moved : ∀ g y, y < (B.opening g).cells (B.fold g) →
      (B.ofFlat table).flat (B.relocate g y) = table (B.relocate g y) := fun g y inside => by
    rw [Witness.flat_relocate, ofFlat_group, (B.opening g).flat_ofFlat (B.fold g) _ inside]
  unfold cells openSecondStart secondStart openFirstStart at small
  by_cases main : x < B.chainStart
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset B.mainCells main
    obtain ⟨first, second, third, last⟩ := B.starts g
    have cell := moved g y (by
      have same : B.mainCells g = (B.opening g).chainStart (B.fold g) := rfl
      omega)
    rw [B.relocate_main g inside] at cell
    exact cell
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt main)
  by_cases one : x < ∑ g, firstCells (B.shape g).chain
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset _ one
    obtain ⟨first, second, third, last⟩ := B.starts g
    have cell := moved g ((B.opening g).chainStart (B.fold g) + y) (by omega)
    rw [B.relocate_first g inside] at cell
    rw [← Nat.add_assoc]
    exact cell
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt one)
  rw [← Nat.add_assoc,
    ← show B.openFirstStart = B.chainStart + ∑ g, firstCells (B.shape g).chain from rfl]
  by_cases two : x < firstCells B.chain
  · rw [Witness.flat_openFirst _ two]
    exact PathDigits.firstTable_ofTables B.chain _ _ x two
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt two)
  rw [← Nat.add_assoc, ← show B.secondStart = B.openFirstStart + firstCells B.chain from rfl]
  by_cases three : x < ∑ g, secondCells (B.shape g).chain
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset _ three
    obtain ⟨first, second, third, last⟩ := B.starts g
    have cell := moved g ((B.opening g).secondStart (B.fold g) + y) (by omega)
    rw [B.relocate_second g inside] at cell
    rw [← Nat.add_assoc]
    exact cell
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt three)
  rw [← Nat.add_assoc,
    ← show B.openSecondStart = B.secondStart + ∑ g, secondCells (B.shape g).chain from rfl,
    Witness.flat_openSecond _ (show x < secondCells B.chain by omega)]
  exact PathDigits.secondTable_ofTables B.chain _ _ x (by omega)

section Rows

variable {E : Type} [Field E] [Algebra Zq E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
  (st : B.Statement E) (challenge : B.Challenge) (μ : ℕ) (fits : B.cells < 2 ^ μ)

/-- The ring rows of `Batch.flatRows`, and the row `w_cells = 0` in `R_{q,d_D}`. -/
noncomputable def lockedRows : Fin (B.rowCount + 1) → RingRow :=
  Fin.append (B.flatRows β st challenge) fun _ : Fin 1 => (zeroForm B.openDegree B.cells).row 0

theorem lockedRows_below (r : Fin (B.rowCount + 1)) :
    (B.lockedRows β st challenge r).Below B.cells := by
  revert r
  exact (forall_append (RingRow.Below · B.cells) _ _).mpr
    ⟨B.flatRows_below β st challenge,
      fun _ => Form.Below.row (f := zeroForm B.openDegree B.cells) (fun _ _ => le_rfl) 0⟩

theorem lockedRows_degree_le (r : Fin (B.rowCount + 1)) :
    (B.lockedRows β st challenge r).degree ≤ B.largestDegree := by
  revert r
  exact (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr
    ⟨B.flatRows_degree_le β st challenge,
      fun _ => le_max_of_le_right (le_max_left B.openDegree (chainDegree B.chain))⟩

theorem lockedRows_degree_sum :
    ∑ r, (B.lockedRows β st challenge r).degree = B.degreeSum + B.openDegree := by
  unfold lockedRows
  rw [sum_append RingRow.degree, B.flatRows_degree_sum]
  rfl

/-- The relation of one fold with opening groups for a table that the next level commits to: the
rows and the scalar row of `Batch.flatRelation`, and the row `w_cells = 0`. -/
noncomputable def lockedRelation : FlatRelation E where
  vars := μ
  ring := B.rowCount + 1
  row := B.lockedRows β st challenge
  inside r c j _ := (B.lockedRows_below β st challenge r c j).trans_lt fits
  weight := B.scalarWeight β st μ
  value := st.value

variable {B β st challenge μ fits}

theorem lockedRelation_holds {table : Table}
    (holds : (B.lockedRelation β st challenge μ fits).Holds table) :
    table B.cells = 0 ∧ (B.flatRelation β st challenge μ fits).Holds table := by
  have rows : ∀ r, (Fin.append (B.flatRows β st challenge)
      (fun _ : Fin 1 => (zeroForm B.openDegree B.cells).row 0) r).Holds table := holds.1
  obtain ⟨flat, zero⟩ := (forall_append (RingRow.Holds · table) (B.flatRows β st challenge)
    (fun _ : Fin 1 => (zeroForm B.openDegree B.cells).row 0)).mp rows
  exact ⟨zero_of_zeroForm (zero 0), flat, holds.2⟩

/-- A table with `Batch.flatRelation` and the cell `cells` equal to 0 satisfies
`Batch.lockedRelation`. -/
theorem lockedRelation_of_flat {table : Table} (zero : table B.cells = 0)
    (holds : (B.flatRelation β st challenge μ fits).Holds table) :
    (B.lockedRelation β st challenge μ fits).Holds table := by
  refine ⟨(forall_append (RingRow.Holds · table) (B.flatRows β st challenge)
    (fun _ : Fin 1 => (zeroForm B.openDegree B.cells).row 0)).mpr ⟨holds.1, fun _ =>
      ((zeroForm B.openDegree B.cells).row_holds 0 table).mpr ?_⟩, holds.2⟩
  show (∑ _c : Unit, (1 : Rq B.openDegree) * ofCoeffs fun _ => table B.cells) = 0
  rw [zero, ofCoeffs_eq (x := 0) fun j _ => (coeff_zero j).symm]
  simp

/-- Two tables with the same cells up to `cells` satisfy `Batch.flatRelation` together. -/
theorem flatRelation_holds_congr {table other : Table} (same : ∀ x ≤ B.cells, table x = other x)
    (holds : (B.flatRelation β st challenge μ fits).Holds table) :
    (B.flatRelation β st challenge μ fits).Holds other := by
  refine ⟨fun r => RingRow.holds_congr (B.flatRows_below β st challenge r) same (holds.1 r), ?_⟩
  rw [← holds.2]
  refine Finset.sum_congr rfl fun u _ => ?_
  show algebraMap Zq E (other u) * B.scalarWeight β st μ u =
    algebraMap Zq E (table u) * B.scalarWeight β st μ u
  by_cases inside : u ≤ B.cells
  · rw [same u inside]
  · rw [B.scalarWeight_of_lt β st μ (Nat.lt_of_not_le inside), mul_zero, mul_zero]

end Rows

section Alphabets

variable {B} {w : B.Witness} {base : ℕ}

/-- Each cell of the flat witness of digit vectors with the certified alphabets is in `A_{b⋆}`,
and each cell from the first digit of `F_1` is in `{−1, 0}` (Akita (76) and Remark 6.2), when
each group has the certified base `b⋆ = base ≥ 2`. -/
theorem Witness.Alphabets.flat_cell (alphabets : w.Alphabets)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (two : 2 ≤ base) (x : ℕ) :
    (-((base / 2 : ℕ) : ℤ) ≤ (w.flat x).valMinAbs ∧
      (w.flat x).valMinAbs ≤ ((base / 2 : ℕ) : ℤ) - 1) ∧
      (B.chainStart ≤ x → -1 ≤ (w.flat x).valMinAbs ∧ (w.flat x).valMinAbs ≤ 0) := by
  have low : -((base / 2 : ℕ) : ℤ) ≤ 0 := by omega
  have high : (0 : ℤ) ≤ ((base / 2 : ℕ) : ℤ) - 1 := by omega
  have binary : ∀ (z : Zq) (late : Prop), -1 ≤ z.valMinAbs ∧ z.valMinAbs ≤ 0 →
      (-((base / 2 : ℕ) : ℤ) ≤ z.valMinAbs ∧ z.valMinAbs ≤ ((base / 2 : ℕ) : ℤ) - 1) ∧
        (late → -1 ≤ z.valMinAbs ∧ z.valMinAbs ≤ 0) :=
    fun z _ range => ⟨⟨by omega, by omega⟩, fun _ => range⟩
  by_cases main : x < B.chainStart
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset B.mainCells main
    have cell := ((alphabets.groups g).flat_cell (by rw [uniform g]; exact two) y).1
    rw [uniform g] at cell
    rw [show offset B.mainCells g + y = B.mainStart g + y from rfl, w.flat_main g inside]
    exact ⟨cell, fun start => absurd start (Nat.not_le.mpr main)⟩
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt main)
  by_cases one : x < ∑ g, firstCells (B.shape g).chain
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset _ one
    rw [show B.chainStart + (offset (fun h => firstCells (B.shape h).chain) g + y) =
      B.firstStart g + y from (Nat.add_assoc _ _ _).symm, w.flat_first g inside]
    exact binary _ _ ((alphabets.groups g).chainDigits.firstTable y)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt one)
  rw [show B.chainStart + (∑ g, firstCells (B.shape g).chain + x) = B.openFirstStart + x from
    (Nat.add_assoc _ _ _).symm]
  by_cases two' : x < firstCells B.chain
  · rw [w.flat_openFirst two']
    exact binary _ _ (alphabets.openDigits.firstTable x)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt two')
  rw [show B.openFirstStart + (firstCells B.chain + x) = B.secondStart + x from
    (Nat.add_assoc _ _ _).symm]
  by_cases three : x < ∑ g, secondCells (B.shape g).chain
  · obtain ⟨g, y, inside, rfl⟩ := exists_offset _ three
    rw [show B.secondStart + (offset (fun h => secondCells (B.shape h).chain) g + y) =
      B.secondStartOf g + y from (Nat.add_assoc _ _ _).symm, w.flat_second g inside]
    exact binary _ _ ((alphabets.groups g).chainDigits.secondTable y)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt three)
  rw [show B.secondStart + (∑ g, secondCells (B.shape g).chain + x) = B.openSecondStart + x from
    (Nat.add_assoc _ _ _).symm]
  by_cases four : x < secondCells B.chain
  · rw [w.flat_openSecond four]
    exact binary _ _ (alphabets.openDigits.secondTable x)
  · rw [w.flat_of_le (show B.cells ≤ B.openSecondStart + x by
      unfold cells
      omega), ZMod.valMinAbs_zero]
    exact ⟨⟨low, high⟩, fun _ => ⟨by decide, le_rfl⟩⟩

/-- The digit range statement of Akita Section 6.1 for the flat witness of digit vectors with
the certified alphabets. -/
theorem Witness.Alphabets.rangeHolds (alphabets : w.Alphabets)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (two : 2 ≤ base)
    (small : base < fieldModulus) (μ : ℕ) : RangeHolds base μ w.flat :=
  (rangeHolds_iff small μ _).mpr fun x _ => (alphabets.flat_cell uniform two x).1

/-- The binary statement of Akita Remark 6.2 for the flat witness of digit vectors with the
certified alphabets. -/
theorem Witness.Alphabets.binaryOn (alphabets : w.Alphabets)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (two : 2 ≤ base) (μ : ℕ) :
    BinaryOn B.binarySupport μ w.flat :=
  fun x _ member =>
    (mul_add_one_eq_zero_iff _).mpr ((alphabets.flat_cell uniform two x).2 member.1)

end Alphabets

section Check

variable {B} {E : Type} [Field E] [Fintype E] [Algebra Zq E]
  (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E) (challenge : B.Challenge)
  (μ : ℕ) (fits : B.cells < 2 ^ μ)

/-- The quotient-free reduction of `Batch.lockedRelation` (Akita Section 7.3 and Theorem 7.4),
with `2^batch` places for the rows. Its bad set has at most `d_max − 1` points. -/
noncomputable def lockedReduction (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) :
    Reduction E :=
  (B.lockedRelation β st challenge μ fits).free (algebraMap Zq E).injective B.largestDegree batch
    (B.lockedRows_degree_le β st challenge) rowsFit

variable {base : ℕ}

omit [Fintype E] in
/-- A table with `Batch.lockedRelation`, the digit range statement and the binary statement has
digit vectors `Batch.ofFlat` with the exact branch relation `Batch.Witness.Satisfies`. Each group
has the certified base `base`. The table can have each value in the cells after `cells`. -/
theorem satisfies_of_locked (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) {table : Table}
    (relation : (B.lockedRelation β st challenge μ fits).Holds table)
    (range : RangeHolds base μ table) (binary : BinaryOn B.binarySupport μ table) :
    (B.ofFlat table).Satisfies β st challenge := by
  obtain ⟨zero, flat⟩ := lockedRelation_holds relation
  have same : ∀ x ≤ B.cells, table x = (B.ofFlat table).flat x := fun x below => by
    rcases Nat.lt_or_eq_of_le below with inside | rfl
    · exact (B.flat_ofFlat table inside).symm
    · rw [zero, Witness.flat_of_le _ le_rfl]
  have ranged : RangeHolds base μ (B.ofFlat table).flat := fun x inside => by
    by_cases below : x ≤ B.cells
    · rw [← same x below]
      exact range x inside
    · have last := range B.cells fits
      rw [zero] at last
      rw [Witness.flat_of_le _ (Nat.le_of_lt (Nat.lt_of_not_le below))]
      exact last
  have binaried : BinaryOn B.binarySupport μ (B.ofFlat table).flat := fun x inside member => by
    rw [← same x (Nat.le_of_lt member.2)]
    exact binary x inside member
  exact Witness.satisfies_of_flat μ fits
    (Witness.alphabets_of_flat uniform small fits ranged binaried)
    (flatRelation_holds_congr same flat)

variable [DecidableEq E]

/-- Akita Lemma 10.42 (i) for the table of a successor source of a fold with opening groups, with
`foldCheck_avoids`: on a path of `foldCheck` that avoids the bad sets, a final claim that is true
for the table gives the exact branch relation `Batch.Witness.Satisfies` of the digit vectors
`Batch.ofFlat` of the table (the hypothesis of Akita Corollary 10.32 for one child). Each group
has the certified base `base`. -/
theorem lockedFold_satisfies (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch)
    (table : Table) :
    (foldCheck (lockedReduction β st challenge μ fits batch rowsFit) base B.binarySupport
      table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (B.ofFlat table).Satisfies β st challenge :=
  (foldCheck_avoids (lockedReduction β st challenge μ fits batch rowsFit) base B.binarySupport
    table).mono fun _ checked r2 v equal opening =>
      satisfies_of_locked β st challenge μ fits uniform small (checked r2 v equal opening).1
        (checked r2 v equal opening).2.1 (checked r2 v equal opening).2.2

variable {β st challenge μ fits} {w : B.Witness} (admits : Type → Prop)

/-- The completeness of the checks of a last fold with opening groups (Akita Lemma 10.7 with
Theorem 10.9): for digit vectors with the exact branch relation, the prover has a message for
each draw sequence, and the checks end in the claim `(r_2, w̃(r_2))` for the flat witness. -/
theorem flatFold_winsTo (scalar : admits E) (range : admits (Fin (base / 2 + 1 + 1) → E))
    (cubic : admits (Fin (3 + 1) → E)) (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (two : 2 ≤ base) (small : base < fieldModulus) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (flatReduction β st challenge μ fits batch rowsFit) base
      B.binarySupport w.flat) (0 + μ + 1 + 1 + μ + 1 + 1 + 1) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheck_winsTo admits (flatReduction β st challenge μ fits batch rowsFit) base scalar range
    cubic _ (satisfies.flat μ fits) (satisfies.alphabets.rangeHolds uniform two small μ)
    (satisfies.alphabets.binaryOn uniform two μ)

/-- `Batch.flatFold_winsTo` for the checks of a fold with opening groups whose flat witness the
next level commits to. -/
theorem lockedFold_winsTo (scalar : admits E) (range : admits (Fin (base / 2 + 1 + 1) → E))
    (cubic : admits (Fin (3 + 1) → E)) (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (two : 2 ≤ base) (small : base < fieldModulus) (batch : ℕ)
    (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) (satisfies : w.Satisfies β st challenge) :
    WinsTo admits (foldCheck (lockedReduction β st challenge μ fits batch rowsFit) base
      B.binarySupport w.flat) (0 + μ + 1 + 1 + μ + 1 + 1 + 1) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheck_winsTo admits (lockedReduction β st challenge μ fits batch rowsFit) base scalar range
    cubic _ (lockedRelation_of_flat (w.flat_of_le le_rfl) (satisfies.flat μ fits))
    (satisfies.alphabets.rangeHolds uniform two small μ)
    (satisfies.alphabets.binaryOn uniform two μ)

end Check

end Batch

end NightstreamFPrime.Spec.Compression.Akita
