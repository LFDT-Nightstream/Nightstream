import Nightstream.Implementation.Lowering.Goldilocks.Rows
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: semantics-preserving translation from the repository's numeric
Goldilocks R1CS rows to stable typed lowering rows.

Assurance tier: model-level.

Owns:
- reduction of every numeric coefficient and assignment coordinate into the
  paper Goldilocks carrier;
- exact translation of sparse terms and rows through an explicit stable
  column map;
- equivalence of numeric and typed row satisfaction;
- occurrence-preserving row ownership, duplicate-free row identities, and
  whole-list satisfaction transport.

Does not own: a concrete column map, allocation ownership, a protocol
primitive, activation gating, generated-row equality, Rust semantics, or
Poseidon2 correctness.

Emits constraints: no. It translates a supplied row list without adding or
removing occurrences.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete

namespace Numeric

abbrev Row := Nightstream.Implementation.R1CS.Row

def satisfies :=
  Nightstream.Implementation.R1CS.Satisfies

def rowHolds :=
  Nightstream.Implementation.R1CS.RowHolds

def lcEval :=
  Nightstream.Implementation.R1CS.lcEval

def modulus :=
  Nightstream.Implementation.R1CS.goldilocksP

end Numeric

/-- Canonical interpretation of any natural number in the paper Goldilocks
carrier. -/
def residue (value : Nat) : F :=
  ⟨value % Nightstream.SuperNeo.Concrete.goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

@[simp] theorem residue_zero : residue 0 = (0 : F) :=
  rfl

@[simp] theorem residue_one : residue 1 = (1 : F) :=
  rfl

theorem residue_modulus_eq :
    Numeric.modulus =
      Nightstream.SuperNeo.Concrete.goldilocksModulus :=
  rfl

theorem residue_mod (value : Nat) :
    residue (value % Numeric.modulus) = residue value := by
  apply Fin.ext
  simp [residue, Numeric.modulus,
    Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus]

theorem residue_add (left right : Nat) :
    residue (left + right) = residue left + residue right := by
  apply Fin.ext
  change
    (left + right) %
        Nightstream.SuperNeo.Concrete.goldilocksModulus =
      (left % Nightstream.SuperNeo.Concrete.goldilocksModulus +
          right % Nightstream.SuperNeo.Concrete.goldilocksModulus) %
        Nightstream.SuperNeo.Concrete.goldilocksModulus
  exact Nat.add_mod left right
    Nightstream.SuperNeo.Concrete.goldilocksModulus

theorem residue_mul (left right : Nat) :
    residue (left * right) = residue left * residue right := by
  apply Fin.ext
  change
    (left * right) %
        Nightstream.SuperNeo.Concrete.goldilocksModulus =
      (left % Nightstream.SuperNeo.Concrete.goldilocksModulus *
          (right % Nightstream.SuperNeo.Concrete.goldilocksModulus)) %
        Nightstream.SuperNeo.Concrete.goldilocksModulus
  exact Nat.mul_mod left right
    Nightstream.SuperNeo.Concrete.goldilocksModulus

@[simp] theorem residue_field_val (value : F) :
    residue value.val = value := by
  apply Fin.ext
  simp [residue, Nat.mod_eq_of_lt value.isLt]

theorem residue_injective_of_lt
    {left right : Nat}
    (leftLt : left < Numeric.modulus)
    (rightLt : right < Numeric.modulus)
    (equal : residue left = residue right) :
    left = right := by
  have leftLtConcrete :
      left < Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    simpa [Numeric.modulus,
      Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using leftLt
  have rightLtConcrete :
      right < Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    simpa [Numeric.modulus,
      Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using rightLt
  have values := congrArg Fin.val equal
  simpa only [residue, Nat.mod_eq_of_lt leftLtConcrete,
    Nat.mod_eq_of_lt rightLtConcrete] using values

/-- Pull a typed field assignment back to canonical numeric representatives
through one explicit source-column map. -/
def numericAssignment
    (columnMap : Nat -> ColumnId)
    (assignment : ColumnId -> F) :
    Nat -> Nat :=
  fun source => (assignment (columnMap source)).val

/-- Canonical representatives of an arbitrary numeric assignment. -/
def canonicalAssignment (assignment : Nat → Nat) : Nat → Nat :=
  fun source => assignment source % Numeric.modulus

private theorem rawLcEval_canonical_mod
    (assignment : Nat → Nat) :
    ∀ terms : List (Nat × Nat),
      Nightstream.Implementation.R1CS.Program.rawLcEval
          (canonicalAssignment assignment) terms % Numeric.modulus =
        Nightstream.Implementation.R1CS.Program.rawLcEval
          assignment terms % Numeric.modulus
  | [] => rfl
  | term :: tail => by
      simp only [Nightstream.Implementation.R1CS.Program.rawLcEval,
        canonicalAssignment]
      have termMod :
          term.2 * (assignment term.1 % Numeric.modulus) %
              Numeric.modulus =
            term.2 * assignment term.1 % Numeric.modulus := by
        calc
          term.2 * (assignment term.1 % Numeric.modulus) %
                Numeric.modulus =
              (term.2 % Numeric.modulus *
                  (assignment term.1 % Numeric.modulus) %
                    Numeric.modulus) :=
            by simpa only [Nat.mod_mod] using
              Nat.mul_mod term.2
                (assignment term.1 % Numeric.modulus) Numeric.modulus
          _ = term.2 * assignment term.1 % Numeric.modulus :=
            (Nat.mul_mod _ _ _).symm
      calc
        (term.2 * (assignment term.1 % Numeric.modulus) +
            Nightstream.Implementation.R1CS.Program.rawLcEval
              (canonicalAssignment assignment) tail) % Numeric.modulus =
          (term.2 * (assignment term.1 % Numeric.modulus) %
              Numeric.modulus +
            Nightstream.Implementation.R1CS.Program.rawLcEval
              (canonicalAssignment assignment) tail % Numeric.modulus) %
              Numeric.modulus :=
            Nat.add_mod _ _ _
        _ = (term.2 * assignment term.1 % Numeric.modulus +
            Nightstream.Implementation.R1CS.Program.rawLcEval
              assignment tail % Numeric.modulus) % Numeric.modulus := by
          rw [termMod, rawLcEval_canonical_mod assignment tail]
        _ = (term.2 * assignment term.1 +
            Nightstream.Implementation.R1CS.Program.rawLcEval
              assignment tail) % Numeric.modulus :=
          (Nat.add_mod _ _ _).symm

theorem lcEval_canonical
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    Numeric.lcEval (canonicalAssignment assignment) terms =
      Numeric.lcEval assignment terms := by
  change
    Nightstream.Implementation.R1CS.lcEval
        (canonicalAssignment assignment) terms =
      Nightstream.Implementation.R1CS.lcEval assignment terms
  rw [Nightstream.Implementation.R1CS.Program.lcEval_eq_raw_mod,
    Nightstream.Implementation.R1CS.Program.lcEval_eq_raw_mod]
  change
    Nightstream.Implementation.R1CS.Program.rawLcEval
          (canonicalAssignment assignment) terms % Numeric.modulus =
      Nightstream.Implementation.R1CS.Program.rawLcEval
          assignment terms % Numeric.modulus
  exact rawLcEval_canonical_mod assignment terms

theorem satisfies_canonical
    (rows : List Numeric.Row) (assignment : Nat → Nat)
    (satisfied : Numeric.satisfies rows assignment) :
    Numeric.satisfies rows (canonicalAssignment assignment) := by
  intro row member
  have holds := satisfied row member
  change Nightstream.Implementation.R1CS.RowHolds
    (canonicalAssignment assignment) row
  change Nightstream.Implementation.R1CS.RowHolds assignment row at holds
  unfold Nightstream.Implementation.R1CS.RowHolds at holds ⊢
  have aEqual :
      Nightstream.Implementation.R1CS.lcEval
          (canonicalAssignment assignment) row.a =
        Nightstream.Implementation.R1CS.lcEval assignment row.a :=
    lcEval_canonical assignment row.a
  have bEqual :
      Nightstream.Implementation.R1CS.lcEval
          (canonicalAssignment assignment) row.b =
        Nightstream.Implementation.R1CS.lcEval assignment row.b :=
    lcEval_canonical assignment row.b
  have cEqual :
      Nightstream.Implementation.R1CS.lcEval
          (canonicalAssignment assignment) row.c =
        Nightstream.Implementation.R1CS.lcEval assignment row.c :=
    lcEval_canonical assignment row.c
  rw [aEqual, bEqual, cEqual]
  exact holds

theorem numericAssignment_canonical
    (columnMap : Nat -> ColumnId)
    (assignment : ColumnId -> F)
    (source : Nat) :
    numericAssignment columnMap assignment source < Numeric.modulus := by
  exact (assignment (columnMap source)).isLt

/-- Translate one numeric coefficient-column pair without changing its
source order. -/
def term
    (columnMap : Nat -> ColumnId)
    (source : Nat × Nat) :
    Term where
  column := columnMap source.1
  coefficient := residue source.2

def terms
    (columnMap : Nat -> ColumnId)
    (source : List (Nat × Nat)) :
    LinearCombination :=
  source.map (term columnMap)

private theorem residue_rawLcEval
    (columnMap : Nat -> ColumnId)
    (assignment : ColumnId -> F)
    (source : List (Nat × Nat)) :
    residue
        (Nightstream.Implementation.R1CS.Program.rawLcEval
          (numericAssignment columnMap assignment) source) =
      (terms columnMap source).eval assignment := by
  induction source with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [
        Nightstream.Implementation.R1CS.Program.rawLcEval,
        terms, List.map_cons, LinearCombination.eval, term
      ]
      rw [residue_add, residue_mul, inductionHypothesis]
      simp only [numericAssignment, residue_field_val, terms]

/-- Sparse typed evaluation is exactly the residue of the repository's
numeric `lcEval` on the pulled assignment. -/
theorem terms_eval_eq_residue_lcEval
    (columnMap : Nat -> ColumnId)
    (assignment : ColumnId -> F)
    (source : List (Nat × Nat)) :
    (terms columnMap source).eval assignment =
      residue
        (Numeric.lcEval
          (numericAssignment columnMap assignment) source) := by
  unfold Numeric.lcEval
  rw [Nightstream.Implementation.R1CS.Program.lcEval_eq_raw_mod]
  change
    (terms columnMap source).eval assignment =
      residue
        (Nightstream.Implementation.R1CS.Program.rawLcEval
          (numericAssignment columnMap assignment) source %
            Numeric.modulus)
  rw [residue_mod]
  exact (residue_rawLcEval columnMap assignment source).symm

/-- Translate one complete sparse equation through the same stable column
map. -/
def row
    (columnMap : Nat -> ColumnId)
    (source : Numeric.Row) :
    Row where
  a := terms columnMap source.a
  b := terms columnMap source.b
  c := terms columnMap source.c

theorem row_columnIds
    (columnMap : Nat -> ColumnId)
    (source : Numeric.Row) :
    (row columnMap source).columnIds =
      (source.a ++ source.b ++ source.c).map
        (fun sourceTerm => columnMap sourceTerm.1) := by
  simp [row, Row.columnIds, terms, term, List.map_append,
    Function.comp_def]

/-- One translated equation holds exactly when the original numeric equation
holds on the canonical representatives of the same typed assignment. -/
theorem row_holds_iff
    (columnMap : Nat -> ColumnId)
    (assignment : ColumnId -> F)
    (source : Numeric.Row) :
    (row columnMap source).Holds assignment ↔
      Numeric.rowHolds
        (numericAssignment columnMap assignment) source := by
  let pulled := numericAssignment columnMap assignment
  let left := Numeric.lcEval pulled source.a
  let right := Numeric.lcEval pulled source.b
  let output := Numeric.lcEval pulled source.c
  have outputLt : output < Numeric.modulus := by
    unfold output Numeric.lcEval Nightstream.Implementation.R1CS.lcEval
    exact Nat.mod_lt _ (by decide)
  have productLt : left * right % Numeric.modulus < Numeric.modulus :=
    Nat.mod_lt _ (by decide)
  change
    (terms columnMap source.a).eval assignment *
          (terms columnMap source.b).eval assignment =
        (terms columnMap source.c).eval assignment ↔
      left * right % Numeric.modulus = output
  rw [
    terms_eval_eq_residue_lcEval,
    terms_eval_eq_residue_lcEval,
    terms_eval_eq_residue_lcEval
  ]
  constructor
  · intro equal
    apply residue_injective_of_lt productLt outputLt
    calc
      residue (left * right % Numeric.modulus) =
          residue (left * right) :=
        residue_mod (left * right)
      _ = residue left * residue right :=
        residue_mul left right
      _ = residue output :=
        equal
  · intro equal
    calc
      residue left * residue right =
          residue (left * right) :=
        (residue_mul left right).symm
      _ = residue (left * right % Numeric.modulus) :=
        (residue_mod (left * right)).symm
      _ = residue output :=
        congrArg residue equal

/-- Preserve every supplied numeric row occurrence while assigning stable
owner-local row ordinals. -/
def ownedRowsFrom
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId) :
    List Numeric.Row -> List OwnedRow
  | [] => []
  | source :: tail =>
      {
        id := { owner := owner, ordinal := firstOrdinal }
        row := row columnMap source
      } ::
        ownedRowsFrom owner (firstOrdinal + 1) columnMap tail

@[simp] theorem ownedRowsFrom_length
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row) :
    (ownedRowsFrom owner firstOrdinal columnMap source).length =
      source.length := by
  induction source generalizing firstOrdinal with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [ownedRowsFrom, inductionHypothesis]

theorem ownedRowsFrom_rows
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row) :
    (ownedRowsFrom owner firstOrdinal columnMap source).map
        (fun owned => owned.row) =
      source.map (row columnMap) := by
  induction source generalizing firstOrdinal with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [ownedRowsFrom, inductionHypothesis]

/-- Row identities are exactly the caller-selected contiguous owner-local
ordinal interval, in source occurrence order. -/
theorem ownedRowsFrom_ids_exact
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row) :
    (ownedRowsFrom owner firstOrdinal columnMap source).map
        (fun owned => owned.id) =
      (List.range' firstOrdinal source.length).map
        (fun ordinal => { owner := owner, ordinal := ordinal }) := by
  induction source generalizing firstOrdinal with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [ownedRowsFrom, List.map_cons, List.length_cons,
        List.range'_succ]
      rw [inductionHypothesis (firstOrdinal := firstOrdinal + 1)]

theorem ownedRowsFrom_owned
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row)
    (owned : OwnedRow)
    (member :
      owned ∈ ownedRowsFrom owner firstOrdinal columnMap source) :
    owned.id.owner = owner := by
  induction source generalizing firstOrdinal with
  | nil =>
      simp [ownedRowsFrom] at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with equal | tailMember
      · subst owned
        rfl
      · exact inductionHypothesis (firstOrdinal := firstOrdinal + 1)
          tailMember

private theorem ownedRowsFrom_ordinal_lower_bound
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row)
    (owned : OwnedRow)
    (member :
      owned ∈ ownedRowsFrom owner firstOrdinal columnMap source) :
    firstOrdinal ≤ owned.id.ordinal := by
  induction source generalizing firstOrdinal with
  | nil =>
      simp [ownedRowsFrom] at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with equal | tailMember
      · subst owned
        exact Nat.le_refl _
      · exact Nat.le_trans (Nat.le_succ firstOrdinal)
          (inductionHypothesis (firstOrdinal := firstOrdinal + 1)
            tailMember)

theorem ownedRowsFrom_ids_nodup
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row) :
    ((ownedRowsFrom owner firstOrdinal columnMap source).map
      (fun owned => owned.id)).Nodup := by
  induction source generalizing firstOrdinal with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      rw [ownedRowsFrom, List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.mp member with ⟨owned, ownedMember, equal⟩
        have lower :=
          ownedRowsFrom_ordinal_lower_bound owner (firstOrdinal + 1)
            columnMap tail owned ownedMember
        have ordinalEqual := congrArg RowId.ordinal equal
        simp only at ordinalEqual
        omega
      · exact inductionHypothesis (firstOrdinal := firstOrdinal + 1)

/-- Whole-list typed satisfaction is equivalent to numeric satisfaction on
the pulled canonical assignment. -/
theorem ownedRowsFrom_satisfies_iff
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (columnMap : Nat -> ColumnId)
    (source : List Numeric.Row)
    (assignment : ColumnId -> F) :
    Satisfies
        (ownedRowsFrom owner firstOrdinal columnMap source)
        assignment ↔
      Numeric.satisfies source
        (numericAssignment columnMap assignment) := by
  induction source generalizing firstOrdinal with
  | nil =>
      simp [ownedRowsFrom, Numeric.satisfies,
        Nightstream.Implementation.R1CS.Satisfies]
  | cons head tail inductionHypothesis =>
      rw [ownedRowsFrom, satisfies_cons,
        row_holds_iff, inductionHypothesis]
      simp only [Numeric.satisfies,
        Nightstream.Implementation.R1CS.Satisfies,
        List.mem_cons]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ candidate member
        rcases member with rfl | tailMember
        · exact headHolds
        · exact tailHolds candidate tailMember
      · intro all
        exact ⟨
          all head (Or.inl rfl),
          fun candidate member => all candidate (Or.inr member)
        ⟩

end Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
