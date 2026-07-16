import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.AggregateAcceptanceRows

/-!
Owns: the typed executable schema consumed by the generated active
aggregate-acceptance Rust artifact.

Does not own: generated data, semantic equivalence proofs, selector
materialization, `ChunkBitOuterImage`, or the fixed-F' 960-role bridge.

Emits constraints: no.

Authority boundary: source rows and active CCS rows are evaluated as explicit
equations. Geometry is audit data only, and no digest authorizes a row.

| Schema branch | Mathematical object | Evaluator |
|---|---|---|
| `SourceRow` | one source equation `A * B = C` | `SourceRow.Holds` |
| `CanonicalInverseDecoder` | deterministic inverse image | `CanonicalInverseDecoder.Holds` |
| `ActiveRow` | role-indexed matrix linear combinations | `ActiveRow.point` |
| `PolynomialTerm` | one sparse CCS monomial | `evalPolynomial` |
| `ChunkGeometry` | exact global source/encoded ownership | data only |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifact

/-- Exact source-language roles for one acceptance chunk. -/
inductive SourceRole where
  | one
  | chunkBit (index : Nat)
  | accept
  | inverse
deriving DecidableEq, Repr

/-- Exact encoded-coordinate roles used by the nine active rows. -/
inductive CoordinateRole where
  | one
  | chunkBit (index : Nat)
  | accept
  | treeOutput (index : Nat)
deriving DecidableEq, Repr

/-- Production arity-48 matrix roles occupied by aggregate acceptance. -/
inductive MatrixRole where
  | selector
  | productLeft (index : Nat)
  | productRight (index : Nat)
  | productOut
  | quadraticBitLeft
  | quadraticBitRight
deriving DecidableEq, Repr

/-- One normalized linear term with a signed Goldilocks coefficient. -/
structure LinearTerm (Role : Type) where
  role : Role
  coefficient : Int
deriving DecidableEq, Repr

abbrev SourceLinearCombination := List (LinearTerm SourceRole)
abbrev CoordinateLinearCombination := List (LinearTerm CoordinateRole)

/-- One exact source R1CS row. -/
structure SourceRow where
  a : SourceLinearCombination
  b : SourceLinearCombination
  c : SourceLinearCombination
deriving DecidableEq, Repr

/-- Canonical projected inverse definition and the source rows owning it. -/
structure CanonicalInverseDecoder where
  output : SourceRole
  difference : SourceLinearCombination
  ownedRowOffsets : List Nat
deriving DecidableEq, Repr

/-- Exact global ownership record for one traced chunk. -/
structure ChunkGeometry where
  sourceRowStart : Nat
  sourceRowEnd : Nat
  sourceColumnStart : Nat
  sourceColumnEnd : Nat
  sourceInputColumns : List Nat
  sourceAcceptColumn : Nat
  sourceInverseColumn : Nat
  encodedInputColumns : List Nat
  encodedAcceptanceColumns : List Nat
  activeRowStart : Nat
  activeRowEnd : Nat
deriving DecidableEq, Repr

/-- One role-to-production-matrix binding. -/
structure MatrixBinding where
  role : MatrixRole
  index : Nat
deriving DecidableEq, Repr

/-- One matrix value represented by a coordinate linear combination. -/
structure MatrixLinearCombination where
  role : MatrixRole
  terms : CoordinateLinearCombination
deriving DecidableEq, Repr

/-- Every nonzero matrix value on one materialized CCS row. -/
abbrev ActiveRow := List MatrixLinearCombination

/-- One exponent in a sparse polynomial monomial. -/
structure VariablePower where
  role : MatrixRole
  power : Nat
deriving DecidableEq, Repr

/-- One signed sparse-polynomial term. -/
structure PolynomialTerm where
  coefficient : Int
  powers : List VariablePower
deriving DecidableEq, Repr

abbrev SourceAssignment := SourceRole → F
abbrev CoordinateAssignment := CoordinateRole → F
abbrev MatrixPoint := MatrixRole → F

/-- Interpret a signed generated coefficient in Goldilocks. -/
def coefficient (value : Int) : F := value

/-- Evaluate one normalized linear combination directly. -/
def evalLinearCombination
    {Role : Type} (assignment : Role → F)
    (terms : List (LinearTerm Role)) : F :=
  (terms.map fun term => coefficient term.coefficient * assignment term.role).sum

/-- Direct source-row acceptance. -/
def SourceRow.Holds (row : SourceRow) (assignment : SourceAssignment) : Prop :=
  evalLinearCombination assignment row.a *
      evalLinearCombination assignment row.b =
    evalLinearCombination assignment row.c

/-- The canonical decoder selects zero exactly on the zero difference. -/
def CanonicalInverseDecoder.Holds
    (decoder : CanonicalInverseDecoder) (assignment : SourceAssignment) : Prop :=
  let difference := evalLinearCombination assignment decoder.difference
  assignment decoder.output = if difference = 0 then 0 else difference⁻¹

/-- Value contributed by one matrix linear combination. -/
def MatrixLinearCombination.value
    (combination : MatrixLinearCombination)
    (coordinates : CoordinateAssignment) : F :=
  evalLinearCombination coordinates combination.terms

/-- Sum the explicitly listed contributions for one matrix role. -/
def ActiveRow.point
    (row : ActiveRow) (coordinates : CoordinateAssignment) : MatrixPoint :=
  fun role =>
    (row.map fun combination =>
      if combination.role = role then combination.value coordinates else 0).sum

/-- Evaluate one sparse CCS monomial. -/
def evalPolynomialTerm (point : MatrixPoint) (term : PolynomialTerm) : F :=
  coefficient term.coefficient *
    (term.powers.map fun factor => point factor.role ^ factor.power).prod

/-- Evaluate the exact sparse CCS specialization. -/
def evalPolynomial (terms : List PolynomialTerm) (point : MatrixPoint) : F :=
  (terms.map (evalPolynomialTerm point)).sum

/-- Direct acceptance of one generated active row. -/
def ActiveRow.Holds
    (terms : List PolynomialTerm) (coordinates : CoordinateAssignment)
    (row : ActiveRow) : Prop :=
  evalPolynomial terms (row.point coordinates) = 0

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifact
