import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.Exactness

/-!
Direct evaluator for the role-normalized aggregate-acceptance artifact.

Owns: signed-coefficient interpretation, coordinate linear combinations,
matrix points, sparse-polynomial evaluation, and the canonical role assignment
for one candidate's bits, tree outputs, and accept value.

Does not own: generated data, semantic equivalence, matrix-index correctness,
physical row placement, selectors, the 960-chunk image, or row removal.

Emits constraints: no.

Authority boundary: this evaluator is handwritten independently of generated
row contents. A generated row becomes meaningful only through refinement to
`AggregateAcceptanceRows`.

| Stage path | Evaluated object | Result type |
|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs` | coordinate and matrix LCs | `GateField` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate` | product-slot polynomial | `GateField` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | root/output polynomial | `GateField` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifact

open Nightstream.Implementation.R1CS
open Mod5

abbrev CoordinateAssignment := CoordinateRole → GateField
abbrev MatrixPoint := MatrixRole → GateField

/-- Interpret a signed generated coefficient in the active Goldilocks carrier. -/
def coefficient : Int → GateField
  | .ofNat value => fieldResidue value
  | .negSucc predecessor =>
      fieldResidue (goldilocksP - ((predecessor + 1) % goldilocksP))

/-- Evaluate one normalized coordinate linear combination. -/
def evalLinearCombination
    (assignment : CoordinateAssignment)
    (terms : CoordinateLinearCombination) : GateField :=
  (terms.map fun term => coefficient term.coefficient * assignment term.role).sum

/-- Value contributed by one matrix linear combination. -/
def MatrixLinearCombination.value
    (combination : MatrixLinearCombination)
    (coordinates : CoordinateAssignment) : GateField :=
  evalLinearCombination coordinates combination.terms

/-- Sum all explicitly listed contributions for one matrix role. -/
def ActiveRow.point
    (row : ActiveRow) (coordinates : CoordinateAssignment) : MatrixPoint :=
  fun role =>
    (row.map fun combination =>
      if combination.role = role then combination.value coordinates else 0).sum

/-- Evaluate the variable powers of one sparse monomial. -/
def evalPowers (point : MatrixPoint) : List VariablePower → GateField
  | [] => 1
  | factor :: tail =>
      point factor.role ^ factor.power * evalPowers point tail

/-- Evaluate one sparse monomial directly in the active Goldilocks carrier. -/
def evalPolynomialTerm (point : MatrixPoint) (term : PolynomialTerm) : GateField :=
  coefficient term.coefficient * evalPowers point term.powers

/-- Evaluate the exact occupied specialization of the production gate. -/
def evalPolynomial (terms : List PolynomialTerm) (point : MatrixPoint) : GateField :=
  (terms.map (evalPolynomialTerm point)).sum

/-- Direct acceptance of one generated role-normalized active row. -/
def ActiveRow.Holds
    (terms : List PolynomialTerm) (coordinates : CoordinateAssignment)
    (row : ActiveRow) : Prop :=
  evalPolynomial terms (row.point coordinates) = 0

/-- Canonical role assignment used to compare generated rows with the
independent aggregate relation. -/
def coordinateAssignment
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : CoordinateAssignment
  | .one => 1
  | .chunkBit index => bits index
  | .accept => accept
  | .treeOutput index => outputs index

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifact
