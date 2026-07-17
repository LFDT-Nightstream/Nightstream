import Nightstream.Implementation.R1CS.Core.Program

/-!
Artifact-owned role language for the aggregate-acceptance leaf.

Owns: finite names for the sixteen input bits, fourteen tree outputs, accept
coordinate, occupied matrix roles, normalized active rows, and sparse gate
terms emitted by the Rust artifact generator.

Does not own: generated production data, mathematical evaluators, sampler
semantics, source-bit decoding, selectors, inactive rows, or physical placement.

Emits constraints: no.

Authority boundary: these types describe generated evidence only. They carry
no claim that a role maps to a production column or satisfies a semantic
obligation.

| Role family | Generated payload | Semantic owner |
|---|---|---|
| coordinates | sixteen source bits, fourteen outputs, accept, one | aggregate artifact refinement |
| matrices | selector, eighteen product pairs, output, Boolean pair | aggregate artifact refinement |
| active rows | seven bit pairs, one product aggregate, one root binding | `AggregateAcceptanceRows` |
| polynomial terms | exact occupied specialization of the arity-56 gate | aggregate artifact refinement |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifact

/-- Finite coordinate roles used by one aggregate-acceptance leaf. -/
inductive CoordinateRole where
  | one
  | chunkBit (index : Fin 16)
  | accept
  | treeOutput (index : Fin 14)
deriving DecidableEq, Repr

/-- Production matrix roles occupied by aggregate acceptance. -/
inductive MatrixRole where
  | selector
  | productLeft (index : Fin 18)
  | productRight (index : Fin 18)
  | productOut
  | quadraticBitLeft
  | quadraticBitRight
deriving DecidableEq, Repr

/-- One normalized linear term with a signed Goldilocks coefficient. -/
structure LinearTerm (Role : Type) where
  role : Role
  coefficient : Int
deriving DecidableEq, Repr

abbrev CoordinateLinearCombination := List (LinearTerm CoordinateRole)

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

/-- Every nonzero matrix value on one materialized leaf row. -/
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

/-- Total selector-inclusive degree of one generated monomial. -/
def PolynomialTerm.totalDegree (term : PolynomialTerm) : Nat :=
  (term.powers.map VariablePower.power).sum

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifact
