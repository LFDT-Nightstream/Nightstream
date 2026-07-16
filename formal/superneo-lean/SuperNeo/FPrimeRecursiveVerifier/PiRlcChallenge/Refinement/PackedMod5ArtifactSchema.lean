import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedChunkRows

/-!
Owns: the typed, directly executable schema for the exact packed Mod-5
Rust-to-Lean artifact.

Does not own: generated production data, the semantic packing proof, selector
composition, or transcript authority.

Emits constraints: no.

Authority boundary: artifact lists are evaluated as equations. Digests are not
part of this schema and cannot authorize a row, decoder, or polynomial term.

| Schema branch | Mathematical object | Evaluator |
|---|---|---|
| `SourceRow` | One source R1CS equation `A * B = C` | `SourceRow.Holds` |
| `DecoderDefinition` | One projected linear/product source definition | `DecoderDefinition.Holds` |
| `ActiveRow` | One of the eight packed row kinds | `activeRowPoint` |
| `PolynomialTerm` | One sparse CCS monomial | `evalPolynomial` |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.PackedMod5Artifact

/-- Exact source-language roles for one traced chunk. -/
inductive SourceRole where
  | one
  | chunkBit (index : Nat)
  | index
  | quotient
  | indexProduct (index : Nat)
  | quotientBit (index : Nat)
deriving DecidableEq, Repr

/-- The fifteen committed coordinates of the packed lowering. -/
inductive CoordinateRole where
  | quotientLow (index : Nat)
  | residueLeft
  | residueRight
deriving DecidableEq, Repr

/-- An atom read by a projected decoder definition. -/
inductive DecoderAtom where
  | source (role : SourceRole)
  | coordinate (role : CoordinateRole)
deriving DecidableEq, Repr

/-- One normalized field-linear term. Coefficients use signed representatives. -/
structure LinearTerm (Role : Type) where
  role : Role
  coefficient : Int
deriving DecidableEq, Repr

abbrev SourceLinearCombination := List (LinearTerm SourceRole)
abbrev DecoderLinearCombination := List (LinearTerm DecoderAtom)

/-- One exact source R1CS row. -/
structure SourceRow where
  a : SourceLinearCombination
  b : SourceLinearCombination
  c : SourceLinearCombination
deriving DecidableEq, Repr

/-- The six projected definitions: three linear values and three products. -/
inductive DecoderDefinition where
  | linear (output : SourceRole) (rhs : DecoderLinearCombination)
  | product (output : SourceRole)
      (left right : DecoderLinearCombination)
deriving DecidableEq, Repr

/-- Roles occupied by the Mod-5 specialization of the production CCS polynomial. -/
inductive MatrixRole where
  | selector
  | bitLeft
  | bitRight
  | residueLeft
  | residueRight
deriving DecidableEq, Repr

/-- One role-to-production-matrix binding. -/
structure MatrixBinding where
  role : MatrixRole
  index : Nat
deriving DecidableEq, Repr

/-- One exponent in a sparse polynomial term. -/
structure VariablePower where
  role : MatrixRole
  power : Nat
deriving DecidableEq, Repr

/-- One sparse term of the exact Mod-5 gate specialization. -/
structure PolynomialTerm where
  coefficient : Int
  powers : List VariablePower
deriving DecidableEq, Repr

/-- Operand roles for the seven packed bit-pair rows. -/
inductive BitOperand where
  | quotientLow (index : Nat)
  | quotientHigh
deriving DecidableEq, Repr

/-- The exact eight-row schedule. -/
inductive ActiveRow where
  | bitPair (left right : BitOperand)
  | residuePair
deriving DecidableEq, Repr

/-- One canonical image of an unsigned residue in the two centered cells. -/
structure CanonicalResidue where
  index : Nat
  left : Int
  right : Int
deriving DecidableEq, Repr

abbrev SourceAssignment := SourceRole → F
abbrev CoordinateAssignment := CoordinateRole → F
abbrev MatrixPoint := MatrixRole → F

/-- Interpret a signed artifact coefficient in Goldilocks. -/
def coefficient (value : Int) : F := value

/-- Direct evaluation of a normalized linear combination. -/
def evalLinearCombination
    {Role : Type} (assignment : Role → F)
    (terms : List (LinearTerm Role)) : F :=
  (terms.map fun term => coefficient term.coefficient * assignment term.role).sum

/-- Direct source-row acceptance. -/
def SourceRow.Holds (row : SourceRow) (assignment : SourceAssignment) : Prop :=
  evalLinearCombination assignment row.a *
      evalLinearCombination assignment row.b =
    evalLinearCombination assignment row.c

/-- A decoder atom reads either a source role or a committed coordinate. -/
def DecoderAtom.value
    (source : SourceAssignment) (coordinates : CoordinateAssignment) :
    DecoderAtom → F
  | .source role => source role
  | .coordinate role => coordinates role

/-- Direct acceptance of one projected decoder definition. -/
def DecoderDefinition.Holds
    (definition : DecoderDefinition)
    (source : SourceAssignment) (coordinates : CoordinateAssignment) : Prop :=
  match definition with
  | .linear output rhs =>
      source output =
        evalLinearCombination (DecoderAtom.value source coordinates) rhs
  | .product output left right =>
      source output =
        evalLinearCombination (DecoderAtom.value source coordinates) left *
          evalLinearCombination (DecoderAtom.value source coordinates) right

/-- Direct sparse-polynomial evaluation. -/
def evalPolynomialTerm (point : MatrixPoint) (term : PolynomialTerm) : F :=
  coefficient term.coefficient *
    (term.powers.map fun factor => point factor.role ^ factor.power).prod

def evalPolynomial (terms : List PolynomialTerm) (point : MatrixPoint) : F :=
  (terms.map (evalPolynomialTerm point)).sum

/-- Interpret one packed bit operand from the semantic witness. -/
noncomputable def bitOperandValue
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : BitOperand → F
  | .quotientLow index =>
      if h : index < 13 then witness.quotientLow ⟨index, h⟩ else 0
  | .quotientHigh => derivedQuotientHighField chunk witness

/-- Matrix values placed on one active packed row. -/
noncomputable def activeRowPoint
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : ActiveRow → MatrixPoint
  | .bitPair left right => fun role =>
      match role with
      | .selector => 1
      | .bitLeft => bitOperandValue chunk witness left
      | .bitRight => bitOperandValue chunk witness right
      | .residueLeft | .residueRight => 0
  | .residuePair => fun role =>
      match role with
      | .selector => 1
      | .bitLeft | .bitRight => 0
      | .residueLeft => witness.residueLeft
      | .residueRight => witness.residueRight

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.PackedMod5Artifact
