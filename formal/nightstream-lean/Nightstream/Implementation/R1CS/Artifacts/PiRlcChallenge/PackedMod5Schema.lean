import Nightstream.Implementation.R1CS.Core.Program

/-!
Artifact-owned role language for the packed Mod-5 sampler leaf.

Owns: finite names for source cells, projected coordinates, decoder atoms,
matrix roles, sparse polynomial terms, and the active row schedule.

Does not own: production data, `ChunkRows` column maps, evaluators, semantic
packing predicates, or refinement theorems.

Emits constraints: no.

Authority boundary: these types describe generated data only. They carry no
claim that a role maps to a particular R1CS column or mathematical equation.

| Role family | Generated payload | Semantic owner |
|---|---|---|
| source and decoder roles | source rows and projected definitions | `Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactSchema` |
| matrix and polynomial roles | arity-56 sparse gate | `Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactRefinement` |
| active-row roles | six low pairs, one high pair, one residue pair | physical-row bridge remains open |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5Artifact

/-- Exact source-language roles for one candidate. Finite indices reject
out-of-profile generated data during elaboration. -/
inductive SourceRole where
  | one
  | chunkBit (index : Fin 16)
  | index
  | quotient
  | indexProduct (index : Fin 3)
  | quotientBit (index : Fin 14)
deriving DecidableEq, Repr

/-- The fifteen committed coordinates used by one packed lowering. -/
inductive CoordinateRole where
  | quotientLow (index : Fin 13)
  | residueLeft
  | residueRight
deriving DecidableEq, Repr

/-- One atom read by a projected decoder definition. -/
inductive DecoderAtom where
  | source (role : SourceRole)
  | coordinate (role : CoordinateRole)
deriving DecidableEq, Repr

/-- One normalized role-linear term. Coefficients retain the signed
representative printed by the Rust artifact generator. -/
structure LinearTerm (Role : Type) where
  role : Role
  coefficient : Int
deriving DecidableEq, Repr

abbrev SourceLinearCombination := List (LinearTerm SourceRole)
abbrev DecoderLinearCombination := List (LinearTerm DecoderAtom)

/-- One exact role-normalized source row. -/
structure SourceRow where
  a : SourceLinearCombination
  b : SourceLinearCombination
  c : SourceLinearCombination
deriving DecidableEq, Repr

/-- Three linear projections followed by the three source product
definitions. -/
inductive DecoderDefinition where
  | linear (output : SourceRole) (rhs : DecoderLinearCombination)
  | product (output : SourceRole)
      (left right : DecoderLinearCombination)
deriving DecidableEq, Repr

/-- Roles occupied by the packed Mod-5 specialization of the production CCS
polynomial. -/
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

/-- One sparse term of the exact packed Mod-5 gate specialization. -/
structure PolynomialTerm where
  coefficient : Int
  powers : List VariablePower
deriving DecidableEq, Repr

/-- Total sparse-monomial degree, including the selector exponent. -/
def PolynomialTerm.totalDegree (term : PolynomialTerm) : Nat :=
  (term.powers.map VariablePower.power).sum

/-- Operand roles for the seven packed bit-pair rows. -/
inductive BitOperand where
  | quotientLow (index : Fin 13)
  | quotientHigh
deriving DecidableEq, Repr

/-- The exact eight-row active schedule. -/
inductive ActiveRow where
  | bitPair (left right : BitOperand)
  | residuePair
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5Artifact
